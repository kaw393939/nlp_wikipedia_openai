# app/entity_processing/enhanced_entity_processor.py

import asyncio
import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiofiles
import aiohttp
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse
from cachetools import LRUCache
from aiolimiter import AsyncLimiter
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from rapidfuzz import process, fuzz
from metaphone import doublemetaphone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langdetect import detect
import yake

from app import retry_async
from app.config_manager import ConfigManager
from app.reports import ReportPostProcessor
from app.exceptions import ProcessingError
from app.workers import extract_and_resolve_entities_worker, preprocess_text_worker
from app.mymodels import ModelType
from dataclasses import dataclass, field


@dataclass
class ProcessingResult:
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


@dataclass
class StoryAnalysisResult:
    """
    Data class to hold all analysis results for a story.
    """
    story_id: str
    language: str
    corrected_content: str
    entities: List[Dict[str, Any]]
    wiki_info: Dict[str, Any]
    sentiment: str
    concepts: List[str]
    emotions: List[str]
    keywords: List[str]
    relations: Any
    summary: str
    embedding: List[float]
    analysis: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class EnhancedEntityProcessor:
    def __init__(
        self,
        openai_client: Any,  # Replace with the correct type for your OpenAI client
        config_manager: ConfigManager,
        loggers: Dict[str, logging.Logger],
    ):
        """
        Initialize the EnhancedEntityProcessor with necessary clients, configurations, and loggers.

        Args:
            openai_client (Any): Asynchronous OpenAI client for LLM interactions.
            config_manager (ConfigManager): Manages configuration settings.
            loggers (Dict[str, logging.Logger]): Dictionary of loggers for different modules.
        """
        self.logger = loggers.get("EnhancedEntityProcessor", logging.getLogger("EnhancedEntityProcessor"))
        self.openai_client = openai_client
        self.config_manager = config_manager
        self.loggers = loggers

        # Initialize LRU caches with configurable max sizes
        cache_config = config_manager.get_cache_config()
        self.wikipedia_cache = LRUCache(maxsize=cache_config.get("wikipedia", {}).get("maxsize", 1000))
        self.wikidata_cache = LRUCache(maxsize=cache_config.get("wikidata", {}).get("maxsize", 1000))

        # Initialize Qdrant client
        qdrant_config = self.config_manager.get_qdrant_config()
        self.qdrant_client = QdrantClient(
            url=qdrant_config.get("url", "http://localhost:6333"),
            api_key=qdrant_config.get("api_key", None),
        )
        self.stories_collection_name = qdrant_config.get("stories_collection", {}).get("name", "stories")
        self.reports_collection_name = qdrant_config.get("reports_collection", {}).get("name", "reports")
        self.vector_size = int(qdrant_config.get("vector_size", 1536))

        # Initialize aiohttp configuration settings
        aiohttp_config = self.config_manager.get_aiohttp_config()
        timeout_seconds = aiohttp_config.get("timeout", 60)
        max_connections = aiohttp_config.get("max_connections", 10)

        # Setup aiohttp connector with connection limits
        connector = aiohttp.TCPConnector(limit=max_connections)
        timeout = aiohttp.ClientTimeout(total=timeout_seconds)

        # Create an aiohttp ClientSession with the specified connector and timeout
        self.session = aiohttp.ClientSession(connector=connector, timeout=timeout)

        # Setup rate limiting using aiolimiter
        rate_limit = aiohttp_config.get("rate_limit", 5)
        self.limiter = AsyncLimiter(max_rate=rate_limit, time_period=1)

        # Initialize retry strategy using tenacity
        retry_config = config_manager.get_retry_config()
        self.retry_strategy = retry(
            reraise=True,
            stop=stop_after_attempt(retry_config.get("retries", 5)),
            wait=wait_exponential(
                multiplier=retry_config.get("base_delay", 0.5),
                min=retry_config.get("base_delay", 0.5),
                max=retry_config.get("max_delay", 10),
            ),
            retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError)),
        )

        # Initialize TF-IDF Vectorizer for context matching
        self.vectorizer = TfidfVectorizer(stop_words="english")

        # Initialize ReportPostProcessor
        self.report_post_processor = ReportPostProcessor(
            self.openai_client, self.config_manager, self.loggers
        )

        # Initialize stories dictionary
        self.stories: Dict[str, str] = {}

    async def __aenter__(self):
        """
        Async context manager entry. Returns the instance itself.
        """
        return self

    async def __aexit__(self, exc_type, exc, tb):
        """
        Async context manager exit. Ensures the aiohttp session is closed.
        """
        await self.close()

    async def close(self):
        """
        Close the aiohttp ClientSession to free up resources.
        """
        if not self.session.closed:
            await self.session.close()
            self.logger.info("aiohttp session closed.")

    @retry
    async def correct_entity_spelling(self, entity: str, context: str) -> Optional[str]:
        """
        Suggest a corrected spelling for the entity based on context using OpenAI's LLM.

        Args:
            entity (str): The potentially misspelled entity name.
            context (str): Contextual information surrounding the entity.

        Returns:
            Optional[str]: The corrected entity name or None if unsuccessful.
        """
        prompt = (
            f"The following entity name may be misspelled based on the context. "
            f"Given the context: \"{context}\", provide the most likely correct spelling for the entity name: \"{entity}\"."
        )
        messages = [
            {
                "role": "system",
                "content": "You are an expert in correcting entity names based on context.",
            },
            {"role": "user", "content": prompt},
        ]
        self.loggers["llm"].debug(f"Messages sent to OpenAI for entity spelling correction: {messages}")

        try:
            model_config = self.config_manager.get_model_config(ModelType.CHAT)
            response = await self.openai_client.chat.completions.create(
                model=model_config["model"],
                messages=messages,
                max_tokens=10,
                temperature=0.0,
            )
            if response.choices and response.choices[0].message:
                corrected = response.choices[0].message.content.strip()
                # Basic validation to avoid irrelevant responses
                if corrected and corrected.lower() != entity.lower():
                    self.loggers["wikipedia"].info(f"Corrected entity name from '{entity}' to '{corrected}'")
                    return corrected
        except Exception as e:
            # Log any exceptions that occur during the OpenAI request
            self.loggers["llm"].error(f"Spell correction failed for '{entity}': {str(e)}")
        return None

    @retry
    async def suggest_alternative_entity_name(self, entity: str, context: str) -> Optional[str]:
        """
        Suggest an alternative name for a given entity using OpenAI's language model, considering the context.

        Args:
            entity (str): The original entity name.
            context (str): Contextual information surrounding the entity.

        Returns:
            Optional[str]: Suggested alternative name or None if unsuccessful.
        """
        # Retrieve prompts from the configuration manager or use default prompts
        prompt_system = self.config_manager.get_prompt("entity_suggestion", "system") or (
            "You are an assistant that suggests alternative names for entities based on context."
        )
        prompt_user_template = self.config_manager.get_prompt("entity_suggestion", "user") or (
            "Given the entity '{entity}' and the context: \"{context}\", suggest alternative names that might be used to find it on Wikipedia."
        )
        prompt_user = prompt_user_template.format(entity=entity, context=context)

        # Prepare messages for the OpenAI chat completion
        messages = [
            {"role": "system", "content": prompt_system},
            {"role": "user", "content": prompt_user},
        ]
        self.loggers["llm"].debug(f"Messages sent to OpenAI for entity suggestion: {messages}")

        try:
            model_config = self.config_manager.get_model_config(ModelType.CHAT)
            response = await self.openai_client.chat.completions.create(
                model=model_config["model"],
                messages=messages,
                max_tokens=50,
                temperature=model_config.get("temperature", 0.7),
            )
            if not response.choices or not response.choices[0].message:
                self.loggers["llm"].error("OpenAI response is missing choices or messages.")
                return None
            # Extract and clean the suggestion from the response
            suggestion = response.choices[0].message.content.strip()
            suggestion = suggestion.split("\n")[0].replace(".", "").strip()
            # Validate the suggestion to avoid irrelevant responses
            if suggestion.lower() in ["hello! how can i assist you today?"]:
                self.loggers["llm"].error(f"Invalid suggestion received from OpenAI: '{suggestion}'")
                return None
            self.loggers["wikipedia"].debug(f"Suggested alternative for '{entity}': '{suggestion}'")
            return suggestion
        except Exception as e:
            # Log any exceptions that occur during the OpenAI request
            self.loggers["llm"].error(f"Failed to suggest alternative entity name for '{entity}': {str(e)}")
            return None

    @retry
    async def get_wikidata_id(self, entity: str, context: str) -> Optional[str]:
        """
        Fetch the Wikidata ID for a given entity using the Wikidata API.
        Integrates both Fuzzy Matching with RapidFuzz and Phonetic Matching with Metaphone.

        Args:
            entity (str): The entity name to search for.
            context (str): Contextual information surrounding the entity.

        Returns:
            Optional[str]: The Wikidata ID if found, else None.
        """
        # Check cache first
        if entity in self.wikidata_cache:
            self.loggers["wikipedia"].debug(f"Found Wikidata ID for '{entity}' in cache.")
            return self.wikidata_cache[entity]

        wikidata_api_url = "https://www.wikidata.org/w/api.php"
        params = {
            "action": "wbsearchentities",
            "format": "json",
            "language": "en",
            "search": entity,
            "type": "item",
            "limit": 20,
        }

        try:
            async with self.limiter:
                async with self.session.get(wikidata_api_url, params=params) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        search_results = data.get("search", [])
                        if search_results:
                            # Exact match
                            wikidata_id = search_results[0].get("id")
                            self.wikidata_cache[entity] = wikidata_id  # Cache the ID
                            self.loggers["wikipedia"].debug(f"Fetched Wikidata ID for '{entity}': {wikidata_id}")
                            return wikidata_id
            # If no exact match found, perform Fuzzy Matching
            return await self.perform_fuzzy_and_phonetic_matching(entity, context)
        except Exception as e:
            # Log any exceptions that occur during the API request
            self.loggers["wikipedia"].error(f"Error fetching Wikidata ID for '{entity}': {e}")
            return None

    @retry
    async def perform_fuzzy_and_phonetic_matching(self, entity: str, context: str) -> Optional[str]:
        """
        Perform Fuzzy and Phonetic matching to find the best Wikidata ID for the given entity.

        Args:
            entity (str): The entity name to search for.
            context (str): Contextual information surrounding the entity.

        Returns:
            Optional[str]: The best matched Wikidata ID or None if not found.
        """
        wikidata_api_url = "https://www.wikidata.org/w/api.php"
        params = {
            "action": "wbsearchentities",
            "format": "json",
            "language": "en",
            "search": entity,
            "type": "item",
            "limit": 20,
        }

        try:
            async with self.limiter:
                async with self.session.get(wikidata_api_url, params=params) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        search_results = data.get("search", [])
                        if not search_results:
                            self.loggers["wikipedia"].warning(f"No search results found for '{entity}' during fuzzy and phonetic matching.")
                            return None

                        # Prepare list of entity names from search results
                        candidate_entities = [result["label"] for result in search_results]

                        # Fuzzy Matching using RapidFuzz
                        best_fuzzy_match, fuzzy_score, _ = process.extractOne(
                            query=entity, choices=candidate_entities, scorer=fuzz.WRatio
                        )
                        self.loggers["wikipedia"].debug(f"Best fuzzy match for '{entity}': '{best_fuzzy_match}' with score {fuzzy_score}")

                        # Define a similarity threshold
                        matching_config = self.config_manager.config.get("matching", {})
                        fuzzy_threshold = matching_config.get("fuzzy_threshold", 80)

                        if fuzzy_score >= fuzzy_threshold:
                            # Find the Wikidata ID for the best fuzzy match
                            for result in search_results:
                                if result["label"] == best_fuzzy_match:
                                    wikidata_id = result["id"]
                                    self.wikidata_cache[entity] = wikidata_id  # Cache the ID
                                    self.loggers["wikipedia"].debug(f"Fuzzy matched Wikidata ID for '{entity}': {wikidata_id}")
                                    return wikidata_id

                        # Phonetic Matching using Metaphone
                        entity_metaphone = doublemetaphone(entity)[0]
                        self.loggers["wikipedia"].debug(f"Metaphone code for '{entity}': {entity_metaphone}")

                        # Find candidates with matching metaphone codes
                        phonetic_candidates = [
                            result
                            for result in search_results
                            if doublemetaphone(result["label"])[0] == entity_metaphone
                        ]

                        if phonetic_candidates:
                            wikidata_id = phonetic_candidates[0]["id"]
                            self.wikidata_cache[entity] = wikidata_id  # Cache the ID
                            self.loggers["wikipedia"].debug(f"Phonetic matched Wikidata ID for '{entity}': {wikidata_id}")
                            return wikidata_id

                        self.loggers["wikipedia"].warning(f"No suitable fuzzy or phonetic match found for '{entity}'.")

                        # Additional Step: Contextual Validation
                        contextual_valid_id = await self.contextual_validate(entity, context, search_results)
                        if contextual_valid_id:
                            return contextual_valid_id

                        return None
            return None
        except Exception as e:
            self.loggers["wikipedia"].error(f"Error during fuzzy and phonetic matching for '{entity}': {e}")
            return None

    async def contextual_validate(
        self, entity: str, context: str, search_results: List[Dict[str, Any]]
    ) -> Optional[str]:
        """
        Validate entities based on context by comparing context embeddings with entity descriptions.

        Args:
            entity (str): The original entity name.
            context (str): Contextual information surrounding the entity.
            search_results (List[Dict[str, Any]]): Search results from Wikidata API.

        Returns:
            Optional[str]: Validated Wikidata ID or None.
        """
        try:
            # Extract descriptions from search results
            descriptions = [result.get("description", "") for result in search_results]
            if not descriptions:
                self.loggers["wikipedia"].warning(f"No descriptions available for contextual validation of '{entity}'.")
                return None

            # Fit the vectorizer on descriptions and context
            self.vectorizer.fit(descriptions + [context])

            # Transform context and descriptions into vectors
            context_vector = self.vectorizer.transform([context])
            description_vectors = self.vectorizer.transform(descriptions)

            # Calculate cosine similarity
            similarities = cosine_similarity(context_vector, description_vectors).flatten()
            self.loggers["wikipedia"].debug(f"Cosine similarities for '{entity}': {similarities}")

            # Find the index with the highest similarity
            best_match_index = similarities.argmax()
            best_similarity = similarities[best_match_index]
            self.loggers["wikipedia"].debug(f"Best contextual similarity for '{entity}': {best_similarity} at index {best_match_index}")

            # Define a similarity threshold for context
            matching_config = self.config_manager.config.get("matching", {})
            context_threshold = matching_config.get("context_threshold", 0.3)

            if best_similarity >= context_threshold:
                best_result = search_results[best_match_index]
                wikidata_id = best_result.get("id")
                self.wikidata_cache[entity] = wikidata_id  # Cache the ID
                self.loggers["wikipedia"].debug(f"Contextually validated Wikidata ID for '{entity}': {wikidata_id}")
                return wikidata_id
            else:
                self.loggers["wikipedia"].warning(f"Contextual similarity below threshold for '{entity}'.")
                return None
        except Exception as e:
            self.loggers["wikipedia"].error(f"Error during contextual validation for '{entity}': {e}")
            raise ProcessingError(f"Contextual validation failed: {e}")

    @retry
    async def validate_entity(self, wikidata_id: str, context: str) -> bool:
        """
        Validate the entity by checking essential properties on Wikidata and ensuring alignment with context.

        Args:
            wikidata_id (str): The Wikidata ID of the entity.
            context (str): Contextual information surrounding the entity.

        Returns:
            bool: True if the entity is valid and aligns with context, False otherwise.
        """
        wikidata_api_url = f"https://www.wikidata.org/wiki/Special:EntityData/{wikidata_id}.json"
        try:
            async with self.limiter:
                async with self.session.get(wikidata_api_url) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        entity_data = data.get("entities", {}).get(wikidata_id, {})
                        claims = entity_data.get("claims", {})
                        descriptions = entity_data.get("descriptions", {})
                        english_description = descriptions.get("en", {}).get("value", "")
                        # Example validation: Check if the entity has an occupation (P106)
                        if "P106" in claims:
                            # Further validate against context
                            context_keywords = set(context.lower().split())
                            description_keywords = set(english_description.lower().split())
                            common_keywords = context_keywords.intersection(description_keywords)
                            if common_keywords:
                                self.loggers["wikipedia"].debug(
                                    f"Entity '{wikidata_id}' aligns with context based on keywords: {common_keywords}"
                                )
                                return True
                            else:
                                self.loggers["wikipedia"].warning(
                                    f"Entity '{wikidata_id}' does not align well with context based on keywords."
                                )
                        else:
                            self.loggers["wikipedia"].warning(
                                f"Entity '{wikidata_id}' lacks essential properties."
                            )
            return False
        except Exception as e:
            # Log any exceptions that occur during the validation
            self.loggers["wikipedia"].error(f"Error validating entity '{wikidata_id}': {e}")
            return False

    @retry
    async def get_entity_sections(self, title: str) -> Dict[str, Any]:
        """
        Fetch specific sections from a Wikipedia page.

        Args:
            title (str): The title of the Wikipedia page.

        Returns:
            Dict[str, Any]: A dictionary of section titles and their corresponding summarized text.
        """
        sections_api_url = f"https://en.wikipedia.org/api/rest_v1/page/mobile-sections/{title}"
        try:
            async with self.limiter:
                async with self.session.get(sections_api_url) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        sections = data.get("sections", [])
                        section_contents = {}
                        for section in sections:
                            section_title = section.get("title", "No Title")
                            section_text = section.get("text", "")
                            if section_text:
                                # Summarize the text
                                summarized_text = await self._summarize_text(section_text)
                                section_contents[section_title] = summarized_text
                        self.loggers["wikipedia"].debug(f"Fetched sections for '{title}': {list(section_contents.keys())}")
                        return section_contents
                    else:
                        # Log an error if the request fails with a non-200 status
                        self.loggers["wikipedia"].error(
                            f"Wikipedia sections request failed for '{title}' with status code {resp.status}"
                        )
                        return {}
        except Exception as e:
            # Log any exceptions that occur during the API request
            self.loggers["wikipedia"].error(f"Exception during Wikipedia sections request for '{title}': {e}")
            return {}

    async def _summarize_text(self, text: str) -> str:
        """
        Summarize the given text using OpenAI's language model.

        Args:
            text (str): The text to be summarized.

        Returns:
            str: The summarized text.
        """
        prompt = (
            "Please provide a concise summary of the following section:\n\n"
            f"{text}"
        )
        messages = [
            {
                "role": "system",
                "content": "You are an assistant that summarizes Wikipedia sections.",
            },
            {"role": "user", "content": prompt},
        ]
        self.loggers["llm"].debug(f"Summarization prompt sent to OpenAI: {prompt}")
        try:
            model_config = self.config_manager.get_model_config(ModelType.CHAT)
            response = await self.openai_client.chat.completions.create(
                model=model_config["model"],
                messages=messages,
                max_tokens=150,
                temperature=0.5,
            )
            if response.choices and response.choices[0].message:
                summary = response.choices[0].message.content.strip()
                self.loggers["llm"].debug(f"Received summary from OpenAI: {summary}")
                return summary
            else:
                self.loggers["llm"].error(
                    "OpenAI response is missing choices or messages during summarization."
                )
                return text  # Fallback to original text if summarization fails
        except Exception as e:
            self.loggers["llm"].error(f"Failed to summarize text: {e}")
            return text  # Fallback to original text in case of exception

    @retry_async()
    async def _get_embedding_async(self, text: str) -> List[float]:
        """
        Generate an embedding for the given text using OpenAI's embedding model.

        Args:
            text (str): The text to generate an embedding for.

        Returns:
            List[float]: The generated embedding vector.
        """
        try:
            model_config = self.config_manager.get_model_config(ModelType.EMBEDDING)
            response = await self.openai_client.embeddings.create(
                input=[text],
                model=model_config["model"],
            )
            if not response.data or not response.data[0].embedding:
                self.loggers["llm"].error("OpenAI response is missing embedding data.")
                raise ProcessingError("Failed to generate embedding due to invalid OpenAI response.")
            embedding = response.data[0].embedding
            self.loggers["llm"].debug(
                f"Generated embedding for text: {text[:30]}... [truncated]. Embedding size: {len(embedding)}"
            )
            return embedding
        except Exception as e:
            self.loggers["llm"].error(f"Failed to generate embedding: {e}")
            raise ProcessingError(f"Failed to generate embedding: {e}")

    def _prepare_qdrant_points(
        self,
        collection_name: str,
        point_id: str,
        content: str,
        wiki_info: Dict[str, Any],
        embedding: List[float],
        story_id: str,
    ) -> List[models.PointStruct]:
        """
        Prepare points for insertion into Qdrant.

        Args:
            collection_name (str): Name of the Qdrant collection.
            point_id: str): Unique identifier for the point.
            content (str): The content to be stored.
            wiki_info (Dict[str, Any]): Wikipedia information associated with the content.
            embedding (List[float]): Embedding vector.
            story_id (str): Identifier for the story.

        Returns:
            List[models.PointStruct]: List containing the prepared point.
        """
        if not isinstance(embedding, list) or len(embedding) != self.vector_size:
            raise ValueError(f"Invalid embedding: expected list of length {self.vector_size}")
        payload = {
            "report_id": point_id,
            "story_id": story_id,
            "content": content,
            "wiki_info": json.dumps(wiki_info),
            "timestamp": datetime.now().isoformat(),
        }
        point = models.PointStruct(
            id=point_id,
            vector=embedding,
            payload=payload,
        )
        self.loggers["main"].debug(
            f"Prepared Qdrant point for ID '{point_id}' in collection '{collection_name}'. Vector size: {len(embedding)}"
        )
        return [point]

    def _generate_report_id(self) -> str:
        """
        Generate a unique report ID using UUID4.

        Returns:
            str: The generated UUID4 string.
        """
        return str(uuid.uuid4())

    def _store_points_in_qdrant_sync(self, collection_name: str, points: List[models.PointStruct]) -> None:
        """
        Synchronously store points in Qdrant.

        Args:
            collection_name (str): Name of the Qdrant collection.
            points (List[models.PointStruct]): List of points to store.

        Raises:
            ProcessingError: If storing points fails.
        """
        try:
            operation_info = self.qdrant_client.upsert(collection_name=collection_name, points=points)
            self.loggers["main"].info(
                f"Stored {len(points)} point(s) in Qdrant collection '{collection_name}'. Operation ID: {operation_info.operation_id}"
            )
            self.loggers["main"].info(f"Upsert operation completed for collection '{collection_name}'")
        except UnexpectedResponse as e:
            raise ProcessingError(f"Failed to store points in Qdrant: {e}")
        except Exception as e:
            raise ProcessingError(f"An unexpected error occurred while storing points in Qdrant: {e}")

    async def _store_points_in_qdrant(self, collection_name: str, points: List[models.PointStruct]) -> None:
        """
        Asynchronously store points in Qdrant by running the synchronous method in an executor.

        Args:
            collection_name (str): Name of the Qdrant collection.
            points (List[models.PointStruct]): List of points to store.
        """
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._store_points_in_qdrant_sync, collection_name, points)

    async def perform_sentiment_analysis(self, text: str) -> str:
        """
        Perform sentiment analysis on the given text.

        Args:
            text (str): The text to analyze.

        Returns:
            str: The sentiment result ('positive', 'negative', or 'neutral').
        """
        prompt = f"Analyze the sentiment of the following text and respond with 'positive', 'negative', or 'neutral':\n\n{text}"
        messages = [{"role": "user", "content": prompt}]
        try:
            model_config = self.config_manager.get_model_config(ModelType.CHAT)
            response = await self.openai_client.chat.completions.create(
                model=model_config["model"],
                messages=messages,
                max_tokens=1,
                temperature=0.0,
            )
            sentiment = response.choices[0].message.content.strip().lower()
            if sentiment in ["positive", "negative", "neutral"]:
                return sentiment
            else:
                self.loggers["llm"].warning(f"Unexpected sentiment result: {sentiment}")
                return "unknown"
        except Exception as e:
            self.loggers["errors"].error(f"Sentiment analysis failed: {e}")
            return "unknown"

    async def extract_concepts(self, text: str) -> List[str]:
        """
        Extract key concepts from the text.

        Args:
            text (str): The text to analyze.

        Returns:
            List[str]: A list of extracted concepts.
        """
        prompt = f"Identify the key concepts in the following text:\n\n{text}\n\nConcepts:"
        messages = [{"role": "user", "content": prompt}]
        try:
            model_config = self.config_manager.get_model_config(ModelType.CHAT)
            response = await self.openai_client.chat.completions.create(
                model=model_config["model"],
                messages=messages,
                max_tokens=50,
                temperature=0.5,
            )
            concepts = response.choices[0].message.content.strip()
            concepts_list = [concept.strip() for concept in concepts.split(",") if concept.strip()]
            return concepts_list
        except Exception as e:
            self.loggers["errors"].error(f"Concept extraction failed: {e}")
            return []

    async def perform_entity_sentiment_analysis(self, entity: str, context: str) -> str:
        """
        Analyze the sentiment associated with a specific entity within the context.

        Args:
            entity (str): The entity to analyze.
            context (str): The full text context.

        Returns:
            str: The sentiment result ('positive', 'negative', or 'neutral').
        """
        prompt = f"In the following text, what is the sentiment towards '{entity}'? Respond with 'positive', 'negative', or 'neutral':\n\n{context}"
        messages = [{"role": "user", "content": prompt}]
        try:
            model_config = self.config_manager.get_model_config(ModelType.CHAT)
            response = await self.openai_client.chat.completions.create(
                model=model_config["model"],
                messages=messages,
                max_tokens=1,
                temperature=0.0,
            )
            sentiment = response.choices[0].message.content.strip().lower()
            if sentiment in ["positive", "negative", "neutral"]:
                return sentiment
            else:
                self.loggers["llm"].warning(f"Unexpected entity sentiment result for '{entity}': {sentiment}")
                return "unknown"
        except Exception as e:
            self.loggers["errors"].error(f"Entity sentiment analysis failed for '{entity}': {e}")
            return "unknown"

    async def extract_entity_relations(self, entities: List[str], text: str) -> Any:
        """
        Extract relationships between entities in the text.

        Args:
            entities (List[str]): A list of entity names.
            text (str): The text to analyze.

        Returns:
            Any: The extracted relationships.
        """
        prompt = f"Extract relationships between the following entities in the text: {', '.join(entities)}.\n\nText:\n{text}\n\nList the relationships:"
        messages = [{"role": "user", "content": prompt}]
        try:
            model_config = self.config_manager.get_model_config(ModelType.CHAT)
            response = await self.openai_client.chat.completions.create(
                model=model_config["model"],
                messages=messages,
                max_tokens=150,
                temperature=0.5,
            )
            relations_text = response.choices[0].message.content.strip()
            return relations_text
        except Exception as e:
            self.loggers["errors"].error(f"Relation extraction failed: {e}")
            return []

    async def perform_emotion_analysis(self, text: str) -> List[str]:
        """
        Analyze the emotions expressed in the text.

        Args:
            text (str): The text to analyze.

        Returns:
            List[str]: A list of detected emotions.
        """
        prompt = f"Identify the emotions expressed in the following text:\n\n{text}\n\nEmotions:"
        messages = [{"role": "user", "content": prompt}]
        try:
            model_config = self.config_manager.get_model_config(ModelType.CHAT)
            response = await self.openai_client.chat.completions.create(
                model=model_config["model"],
                messages=messages,
                max_tokens=50,
                temperature=0.5,
            )
            emotions = response.choices[0].message.content.strip()
            emotions_list = [emotion.strip() for emotion in emotions.split(",") if emotion.strip()]
            return emotions_list
        except Exception as e:
            self.loggers["errors"].error(f"Emotion analysis failed: {e}")
            return []

    def extract_keywords(self, text: str) -> List[str]:
        """
        Extract keywords from the text.

        Args:
            text (str): The text to analyze.

        Returns:
            List[str]: A list of extracted keywords.
        """
        try:
            kw_extractor = yake.KeywordExtractor(lan="en", n=1, top=10)
            keywords = kw_extractor.extract_keywords(text)
            keywords_list = [kw[0] for kw in keywords]
            return keywords_list
        except Exception as e:
            self.loggers["errors"].error(f"Keyword extraction failed: {e}")
            return []

    def detect_language(self, text: str) -> str:
        """
        Detect the language of the given text.

        Args:
            text (str): The text to analyze.

        Returns:
            str: The ISO 639-1 code of the detected language.
        """
        try:
            language = detect(text)
            return language
        except Exception as e:
            self.loggers["errors"].error(f"Language detection failed: {e}")
            return "unknown"

    async def translate_text(self, text: str, target_lang: str = "en") -> str:
        """
        Translate the given text to the target language.

        Args:
            text (str): The text to translate.
            target_lang (str): The target language code.

        Returns:
            str: The translated text.
        """
        # Placeholder for actual translation logic
        self.loggers["main"].info("Translation functionality not implemented. Returning original text.")
        return text  # Assuming text is already in English or translation is not implemented

    def generate_knowledge_graph(self, entities: List[Dict[str, str]], relations: Any) -> None:
        """
        Generate a knowledge graph from entities and their relationships.

        Args:
            entities (List[Dict[str, str]]): List of entity dictionaries.
            relations (Any): The relationships between entities.
        """
        # Placeholder for actual knowledge graph generation logic
        self.loggers["main"].info("Knowledge graph generation not implemented.")

    @retry_async()
    async def process_story_async(self, story_id: str, content: str) -> ProcessingResult:
        """
        Process a single story asynchronously, including preprocessing, entity resolution,
        embedding generation, and storing in Qdrant.

        Args:
            story_id (str): Unique identifier for the story.
            content (str): The content of the story.

        Returns:
            ProcessingResult: Result of the processing, indicating success or failure.
        """
        try:
            # Language detection and translation
            language = self.detect_language(content)
            if language != 'en':
                corrected_content = await self.translate_text(content)
            else:
                corrected_content = content

            self.loggers["main"].debug(f"Starting preprocessing for story '{story_id}'")
            corrected_content = await asyncio.get_running_loop().run_in_executor(
                None, preprocess_text_worker, corrected_content
            )
            self.loggers["main"].debug(f"Preprocessing completed for story '{story_id}'")
            resolved_entities = await asyncio.get_running_loop().run_in_executor(
                None, extract_and_resolve_entities_worker, corrected_content
            )
            self.loggers["main"].debug(f"Entities extracted for story '{story_id}': {resolved_entities}")
            wiki_info = await self.get_entities_info(resolved_entities, corrected_content)

            # Perform additional analyses
            sentiment = await self.perform_sentiment_analysis(corrected_content)
            concepts = await self.extract_concepts(corrected_content)
            emotions = await self.perform_emotion_analysis(corrected_content)
            keywords = self.extract_keywords(corrected_content)
            relations = await self.extract_entity_relations(
                [entity['text'] for entity in resolved_entities], corrected_content
            )

            # Generate analysis using the updated generate_analysis method
            self.loggers["llm"].debug(f"Generating analysis for story '{story_id}'")
            analysis = await self.report_post_processor.generate_analysis(
                corrected_content,
                resolved_entities,
                wiki_info,
                sentiment=sentiment,
                concepts=concepts,
                emotions=emotions,
                keywords=keywords,
                relations=relations,
                summary=None  # Summary is now handled within generate_analysis
            )
            self.loggers["llm"].debug(f"Analysis generated for story '{story_id}'")
            await self._save_intermediary_data_async(
                data=analysis,
                path=f"analysis/{story_id}.json",
                data_type="analysis",
            )

            # Create an instance of StoryAnalysisResult
            result_data = StoryAnalysisResult(
                story_id=story_id,
                language=language,
                corrected_content=corrected_content,
                entities=resolved_entities,
                wiki_info=wiki_info,
                sentiment=sentiment,
                concepts=concepts,
                emotions=emotions,
                keywords=keywords,
                relations=relations,
                summary=analysis,  # Assuming generate_analysis returns a string summary
                embedding=[],  # Placeholder if embedding is handled elsewhere
                analysis=analysis  # Assuming generate_analysis returns a string
            )

            # Generate embedding if required
            self.loggers["llm"].debug(f"Generating embedding for story '{story_id}'")
            embedding = await self._get_embedding_async(corrected_content)
            self.loggers["llm"].debug(f"Embedding generated for story '{story_id}'")
            await self._save_intermediary_data_async(
                data=embedding,
                path=f"embeddings/{story_id}.json",
                data_type="embedding",
            )

            # Store the corrected content and embedding in Qdrant
            story_uuid = self._generate_report_id()
            points = self._prepare_qdrant_points(
                self.stories_collection_name,
                story_uuid,
                corrected_content,
                wiki_info,
                embedding,
                story_id,
            )
            await self._store_points_in_qdrant(
                self.stories_collection_name, points
            )
            self.loggers["main"].debug(f"Stored embedding in Qdrant for story '{story_id}'")

            # Store the refined analysis in Qdrant
            report_id = self._generate_report_id()
            report_embedding = await self._get_embedding_async(result_data.analysis)
            report_points = self._prepare_qdrant_points(
                self.reports_collection_name,
                report_id,
                result_data.analysis,
                wiki_info,
                report_embedding,
                story_id,
            )
            await self._store_points_in_qdrant(
                self.reports_collection_name, report_points
            )
            self.loggers["main"].info(
                f"Refined report for story '{story_id}' stored in '{self.reports_collection_name}' collection."
            )

            # Assign the embedding to the result data
            result_data.embedding = embedding

            self.loggers["main"].info(f"Successfully processed story '{story_id}'")
            return ProcessingResult(success=True, data=result_data)
        except ProcessingError as pe:
            self.loggers["errors"].error(f"Failed to process story '{story_id}': {pe}")
            return ProcessingResult(success=False, error=str(pe))
        except Exception as e:
            self.loggers["errors"].error(f"Failed to process story '{story_id}': {e}")
            return ProcessingResult(success=False, error=str(e))

    async def process_stories_async(
        self, stories_dir: str, output_path: str, summary_output_path: str
    ) -> None:
        """
        Process multiple stories asynchronously, handling concurrency, and generating reports.

        Args:
            stories_dir (str): Directory containing story files.
            output_path (str): Path to save the refined report.
            summary_output_path (str): Path to save the summary report.
        """
        try:
            self._load_stories(stories_dir)
            concurrency_config = self.config_manager.get_concurrency_config()
            semaphore_limit = concurrency_config.get(
                "semaphore_limit", asyncio.cpu_count() * 2
            )
            semaphore = asyncio.Semaphore(semaphore_limit)

            async def semaphore_wrapper(story_id, content):
                async with semaphore:
                    return await self.process_story_async(story_id, content)

            tasks = []
            for story_id, content in self.stories.items():
                self.loggers["main"].info(f"Processing story: {story_id}")
                tasks.append(semaphore_wrapper(story_id, content))
            results = await asyncio.gather(*tasks, return_exceptions=True)
            processed_results = []
            wiki_info = {}
            for result in results:
                if isinstance(result, ProcessingResult):
                    processed_results.append(result)
                    if result.success and result.data:
                        wiki_info[result.data.story_id] = result.data.wiki_info
                elif isinstance(result, Exception):
                    self.loggers["errors"].error(f"Error processing story: {result}")
                    processed_results.append(
                        ProcessingResult(success=False, error=str(result))
                    )
                else:
                    self.loggers["errors"].error(
                        f"Unexpected result type: {type(result)}"
                    )
                    processed_results.append(
                        ProcessingResult(success=False, error="Unexpected result type")
                    )

            # Generate refined report from processed_results
            refined_report = await self.report_post_processor.process_full_report(
                processed_results
            )
            await self._save_report_async(refined_report, output_path)

            # Store refined reports in Qdrant
            for result in processed_results:
                if result.success and result.data:
                    story_id = result.data.story_id
                    report_content = result.data.analysis
                    wiki_info_content = json.dumps(result.data.wiki_info, indent=2)
                    report_id = self._generate_report_id()
                    embedding = await self._get_embedding_async(report_content)
                    report_points = self._prepare_qdrant_points(
                        self.reports_collection_name,
                        report_id,
                        report_content,
                        result.data.wiki_info,
                        embedding,
                        story_id,
                    )
                    await self._store_points_in_qdrant(
                        self.reports_collection_name, report_points
                    )
                    self.loggers["main"].info(
                        f"Refined report for story '{story_id}' stored in '{self.reports_collection_name}' collection."
                    )

            # Generate summary report
            await self.generate_summary_report(summary_output_path)

            self.loggers["main"].info(
                f"Processing complete. Refined reports saved to '{self.reports_collection_name}' and summary saved to {summary_output_path}"
            )
        except Exception as e:
            self.loggers["errors"].error(
                f"Failed to process stories: {e}", exc_info=True
            )
        finally:
            if self.session:
                try:
                    await self.close()
                except Exception as e:
                    self.loggers["errors"].error(
                        f"Failed to close aiohttp session: {e}"
                    )
            if self.qdrant_client:
                try:
                    self.qdrant_client.close()
                    self.loggers["main"].info("Qdrant client connection closed.")
                except Exception as e:
                    self.loggers["errors"].error(
                        f"Failed to close Qdrant client: {e}"
                    )

    def _load_stories(self, stories_dir: str) -> None:
        """
        Load stories from the specified directory.

        Args:
            stories_dir (str): Directory containing story files.

        Raises:
            ProcessingError: If no valid stories are found.
        """
        stories_path = Path(stories_dir)
        if not stories_path.exists() or not stories_path.is_dir():
            raise ProcessingError(f"Stories directory not found: {stories_dir}")
        self.stories = {}
        for story_file in stories_path.glob("*.md"):
            try:
                async def read_file():
                    async with aiofiles.open(
                        story_file, "r", encoding="utf-8", errors="replace"
                    ) as f:
                        return await f.read()

                content = asyncio.run(read_file()).strip()
                if not content:
                    self.loggers["wikipedia"].warning(
                        f"Story file '{story_file}' is empty. Skipping."
                    )
                    continue
                self.stories[story_file.stem] = content
                self.loggers["main"].debug(f"Loaded story '{story_file.stem}'")
            except Exception as e:
                self.loggers["errors"].error(
                    f"Failed to read story file '{story_file}': {e}"
                )
                continue
        if not self.stories:
            raise ProcessingError("No valid stories found to process.")
        self.loggers["main"].info(
            f"Loaded {len(self.stories)} story/stories for processing."
        )

    async def _save_report_async(self, report_content: str, output_path: str) -> None:
        """
        Save the report content to the specified output path asynchronously.

        Args:
            report_content (str): The content of the report.
            output_path (str): Path to save the report.
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            async with aiofiles.open(
                output_file, "w", encoding="utf-8", errors="replace"
            ) as f:
                await f.write(report_content)
            self.loggers["main"].info(f"Report saved to {output_path}")
        except Exception as e:
            self.loggers["errors"].error(
                f"Failed to save report to '{output_path}': {e}"
            )

    async def _save_intermediary_data_async(
        self, data: Any, path: str, data_type: str
    ) -> None:
        """
        Save intermediary data (e.g., embeddings, analysis) to a specified path asynchronously.

        Args:
            data (Any): The data to save.
            path (str): Relative path to save the data.
            data_type (str): Type of data being saved (for logging purposes).
        """
        intermediary_dir = self.config_manager.get_paths_config().get(
            "intermediary_dir", "output/intermediary"
        )
        output_dir = Path(intermediary_dir) / path
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        try:
            if data_type in [
                "entities",
                "wiki_info",
                "embedding",
                "analysis",
                "additional_info",
            ]:
                async with aiofiles.open(
                    output_dir, "w", encoding="utf-8", errors="replace"
                ) as f:
                    await f.write(json.dumps(data, indent=2))
                self.loggers["main"].debug(f"Saved {data_type} for '{path}'")
            elif data_type == "preprocessed_text":
                async with aiofiles.open(
                    output_dir, "w", encoding="utf-8", errors="replace"
                ) as f:
                    await f.write(data)
                self.loggers["main"].debug(f"Saved preprocessed text for '{path}'")
            else:
                self.loggers["main"].warning(
                    f"Unknown data_type '{data_type}' for saving intermediary data."
                )
        except Exception as e:
            self.loggers["errors"].error(
                f"Failed to save intermediary data to '{path}': {e}"
            )

    def _retrieve_all_reports_sync(self, collection_name: str) -> List[str]:
        """
        Synchronously retrieve all reports from a specified Qdrant collection.

        Args:
            collection_name (str): Name of the Qdrant collection.

        Returns:
            List[str]: List of report contents.
        """
        try:
            all_reports = []
            limit = self.config_manager.get_qdrant_config().get("retrieve_limit", 1000)
            scroll_result, _ = self.qdrant_client.scroll(
                collection_name=collection_name,
                limit=limit,
                with_payload=True,
                with_vectors=False,
            )
            if not scroll_result:
                self.loggers["main"].warning(
                    f"No data returned from '{collection_name}' collection."
                )
                return all_reports
            for record in scroll_result:
                if isinstance(record, models.Record) and record.payload:
                    content = record.payload.get("content", "")
                    if content:
                        all_reports.append(content)
                else:
                    self.loggers["errors"].warning(
                        f"Unexpected record type or missing payload: {record}"
                    )
            self.loggers["main"].info(
                f"Retrieved {len(all_reports)} reports from '{collection_name}' collection."
            )
            return all_reports
        except Exception as e:
            self.loggers["errors"].error(
                f"Failed to retrieve reports from '{collection_name}' collection: {e}"
            )
            return []

    async def _retrieve_all_reports_async(self, collection_name: str) -> List[str]:
        """
        Asynchronously retrieve all reports from a specified Qdrant collection.

        Args:
            collection_name (str): Name of the Qdrant collection.

        Returns:
            List[str]: List of report contents.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, self._retrieve_all_reports_sync, collection_name
        )

    async def generate_summary_report(self, summary_output_path: str) -> None:
        """
        Generate a summary report from all stored reports and save it asynchronously.

        Args:
            summary_output_path (str): Path to save the summary report.
        """
        all_reports = await self._retrieve_all_reports_async(
            self.reports_collection_name
        )
        if not all_reports:
            self.loggers["main"].warning(
                "No reports found in the reports collection to generate a summary."
            )
            summary = "No reports available to generate a summary."
        else:
            self.loggers["llm"].debug(
                "Generating summary from all stored reports"
            )
            summary = await self.report_post_processor.generate_summary(all_reports)
        await self._save_report_async(summary, summary_output_path)

    async def get_entities_info(
        self, entities: List[Dict[str, str]], context: str
    ) -> Dict[str, Any]:
        """
        Retrieve comprehensive information about a list of entities.

        Args:
            entities (List[Dict[str, str]]): List of entity dictionaries containing 'text' and 'type' for each entity.
            context (str): Contextual information for processing.

        Returns:
            Dict[str, Any]: Dictionary containing information about each entity.
        """
        wiki_info = {}
        tasks = []
        for entity_dict in entities:
            # Extract the 'text' field from the entity dictionary
            entity = entity_dict.get('text', None)
            if entity:
                tasks.append(self.get_entity_info_single(entity, context))
            else:
                self.loggers['wikipedia'].error(f"Entity dictionary missing 'text' field: {entity_dict}")
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for entity, result in zip([e.get('text') for e in entities if 'text' in e], results):
            if isinstance(result, dict):
                # Perform entity-level sentiment analysis
                entity_sentiment = await self.perform_entity_sentiment_analysis(entity, context)
                result['sentiment'] = entity_sentiment
                wiki_info[entity] = result
            elif isinstance(result, Exception):
                self.loggers['wikipedia'].error(f"Error fetching info for entity '{entity}': {result}")
            else:
                self.loggers['wikipedia'].warning(f"No information found for entity '{entity}'.")
        return wiki_info

    @retry
    async def get_entity_info_single(
        self, entity: str, context: str
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve comprehensive information about a single entity.

        Args:
            entity (str): The entity name to retrieve information for.
            context (str): Contextual information for processing.

        Returns:
            Optional[Dict[str, Any]]: Dictionary containing information about the entity or None if not found.
        """
        try:
            # Step 1: Contextual Spell Correction
            corrected_entity = await self.correct_entity_spelling(entity, context)
            if corrected_entity and corrected_entity != entity:
                self.loggers["wikipedia"].info(
                    f"Corrected entity name from '{entity}' to '{corrected_entity}'"
                )
                entity = corrected_entity

            # Check if the entity information is already cached
            if entity in self.wikipedia_cache:
                self.loggers["wikipedia"].debug(
                    f"Entity '{entity}' found in Wikipedia cache"
                )
                return self.wikipedia_cache[entity]

            self.loggers["wikipedia"].debug(f"Fetching Wikipedia info for entity '{entity}'")
            # Get the Wikidata ID for the entity
            wikidata_id = await self.get_wikidata_id(entity, context)
            if not wikidata_id:
                self.loggers["wikipedia"].warning(f"No Wikidata ID found for '{entity}'.")
                return None

            # Validate the entity using its Wikidata ID
            is_valid = await self.validate_entity(wikidata_id, context)
            if not is_valid:
                self.loggers["wikipedia"].warning(f"Entity '{wikidata_id}' is not valid based on context.")
                return None

            # Fetch Wikidata entity data
            wikidata_api_url = f"https://www.wikidata.org/wiki/Special:EntityData/{wikidata_id}.json"
            async with self.limiter:
                async with self.session.get(wikidata_api_url) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        entities = data.get("entities", {})
                        entity_data = entities.get(wikidata_id, {})
                        claims = entity_data.get("claims", {})
                        descriptions = entity_data.get("descriptions", {})
                        english_description = descriptions.get("en", {}).get("value", "")
                        # Example validation: Check if the entity has an occupation (P106)
                        if "P106" in claims:
                            # Further validate against context
                            context_keywords = set(context.lower().split())
                            description_keywords = set(english_description.lower().split())
                            common_keywords = context_keywords.intersection(description_keywords)
                            if common_keywords:
                                self.loggers["wikipedia"].debug(
                                    f"Entity '{wikidata_id}' aligns with context based on keywords: {common_keywords}"
                                )
                                # Continue processing
                            else:
                                self.loggers["wikipedia"].warning(
                                    f"Entity '{wikidata_id}' does not align well with context based on keywords."
                                )
                                return None
                        else:
                            self.loggers["wikipedia"].warning(
                                f"Entity '{wikidata_id}' lacks essential properties."
                            )
                            return None

                        sitelinks = entity_data.get("sitelinks", {})
                        enwiki = sitelinks.get("enwiki", {})
                        wikipedia_title = enwiki.get("title")
                        if not wikipedia_title:
                            self.loggers["wikipedia"].warning(
                                f"No English Wikipedia page found for '{entity}'."
                            )
                            return None

                        # Fetch Wikipedia summary
                        summary_api_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{wikipedia_title}"
                        async with self.limiter:
                            async with self.session.get(summary_api_url) as summary_resp:
                                if summary_resp.status == 200:
                                    summary_data = await summary_resp.json()
                                    if summary_data.get("type") == "disambiguation":
                                        # Handle disambiguation page
                                        self.loggers["wikipedia"].warning(
                                            f"Entity '{entity}' leads to a disambiguation page."
                                        )
                                        alternative_entity = await self.suggest_alternative_entity_name(
                                            entity, context
                                        )
                                        if alternative_entity and alternative_entity != entity:
                                            self.loggers["wikipedia"].info(
                                                f"Attempting to resolve ambiguity by suggesting alternative entity '{alternative_entity}'"
                                            )
                                            # Recursive call to resolve the alternative entity
                                            return await self.get_entity_info_single(
                                                alternative_entity, context
                                            )
                                        else:
                                            self.loggers["wikipedia"].warning(
                                                f"No suitable alternative could be suggested for disambiguated entity '{entity}'."
                                            )
                                            return None

                                    # Construct the information dictionary with summary details
                                    aliases = []
                                    for lang_aliases in entity_data.get("aliases", {}).values():
                                        aliases.extend(alias.get("value") for alias in lang_aliases)
                                    aliases = list(set(aliases))

                                    categories = []
                                    if "P910" in claims:  # Additional name
                                        categories.extend([claim['mainsnak']['datavalue']['value'] for claim in claims["P910"]])

                                    info = {
                                        "wikidata_id": wikidata_id,
                                        "title": summary_data.get("title", "No title available"),
                                        "summary": summary_data.get("extract", "No summary available"),
                                        "url": summary_data.get("content_urls", {}).get("desktop", {}).get("page", "#"),
                                        "categories": categories,
                                        "type": summary_data.get("type", "UNKNOWN"),
                                        "aliases": aliases if aliases else [entity],
                                        "sentiment": "neutral",  # Placeholder, will be updated later
                                    }

                                    # Fetch and add specific sections from the Wikipedia page
                                    sections = await self.get_entity_sections(wikipedia_title)
                                    relevant_sections = {k: v for k, v in sections.items() if k in ["Early life", "Career", "Personal life"]}
                                    info.update({"sections": relevant_sections})

                                    # Cache the complete information to avoid future redundant API calls
                                    self.wikipedia_cache[entity] = info
                                    self.loggers["wikipedia"].debug(
                                        f"Retrieved and cached Wikipedia info for '{entity}': {info}"
                                    )
                                    return info
                                elif summary_resp.status == 404:
                                    # Handle case where the Wikipedia page is not found
                                    self.loggers["wikipedia"].warning(
                                        f"Wikipedia page not found for '{wikipedia_title}'. Attempting to suggest alternative name."
                                    )
                                    alternative_entity = await self.suggest_alternative_entity_name(
                                        entity, context
                                    )
                                    if alternative_entity and alternative_entity != entity:
                                        self.loggers["wikipedia"].info(
                                            f"Attempting to resolve missing page by suggesting alternative entity '{alternative_entity}'"
                                        )
                                        # Recursive call to resolve the alternative entity
                                        return await self.get_entity_info_single(
                                            alternative_entity, context
                                        )
                                    else:
                                        self.loggers["wikipedia"].warning(
                                            f"No Wikipedia page found for '{entity}' and no alternative could be suggested."
                                        )
                                        return None
                                else:
                                    # Log an error if the summary request fails with a different status code
                                    self.loggers["wikipedia"].error(
                                        f"Wikipedia summary request failed for '{wikipedia_title}' with status code {summary_resp.status}"
                                    )
                                    return None
                    elif resp.status == 404:
                        self.loggers["wikipedia"].warning(
                            f"Wikidata entity not found for '{entity}'."
                        )
                        return None
                    else:
                        self.loggers["wikipedia"].error(
                            f"Wikidata entity request failed for '{entity}' with status code {resp.status}"
                        )
                        return None
        except Exception as e:
            # Log any exceptions that occur during the entire entity information retrieval process
            self.loggers["wikipedia"].error(
                f"Exception during Wikipedia request for '{entity}': {e}"
            )
            return None

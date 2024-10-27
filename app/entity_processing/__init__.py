# app/entity_processing/enhanced_entity_processor.py

import asyncio
import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiofiles  # Asynchronous file operations
import aiohttp  # Asynchronous HTTP client
from qdrant_client import QdrantClient, models  # Client for interacting with Qdrant vector database
from qdrant_client.http.exceptions import UnexpectedResponse  # Exception handling for Qdrant
from cachetools import LRUCache  # Least Recently Used cache for efficient data retrieval
from aiolimiter import AsyncLimiter  # Rate limiter to control the rate of asynchronous operations
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type  # Retry strategy for handling transient failures
from rapidfuzz import process, fuzz  # Fuzzy string matching for entity resolution
from metaphone import doublemetaphone  # Phonetic encoding for matching similar-sounding words
from sklearn.feature_extraction.text import TfidfVectorizer  # TF-IDF vectorizer for text feature extraction
from sklearn.metrics.pairwise import cosine_similarity  # Cosine similarity for comparing text embeddings
from langdetect import detect  # Language detection for text preprocessing
import yake  # Yet Another Keyword Extractor for keyword extraction

from app import retry_async  # Custom decorator for retrying asynchronous functions
from app.config_manager import ConfigManager  # Configuration manager for handling app settings
from app.reports import ReportPostProcessor  # Post-processor for generating and refining reports
from app.exceptions import ProcessingError  # Custom exception for processing-related errors
from app.workers import extract_and_resolve_entities_worker, preprocess_text_worker  # Worker functions for text preprocessing and entity extraction
from app.mymodels import ModelType  # Enum or class defining model types used in the app
from dataclasses import dataclass, field  # Data classes for structured data management

# Define a dataclass to encapsulate the result of processing a story
@dataclass
class ProcessingResult:
    success: bool  # Indicates if processing was successful
    data: Optional[Dict[str, Any]] = None  # Contains processed data if successful
    error: Optional[str] = None  # Contains error message if processing failed

# Define a dataclass to hold all analysis results for a story
@dataclass
class StoryAnalysisResult:
    """
    Data class to hold all analysis results for a story.
    """
    story_id: str  # Unique identifier for the story
    language: str  # Detected language of the story
    corrected_content: str  # Preprocessed and possibly translated content of the story
    entities: List[Dict[str, Any]]  # List of extracted entities from the story
    wiki_info: Dict[str, Any]  # Wikipedia information related to the entities
    sentiment: str  # Overall sentiment of the story
    concepts: List[str]  # Extracted key concepts from the story
    emotions: List[str]  # Detected emotions expressed in the story
    keywords: List[str]  # Extracted keywords from the story
    relations: Any  # Extracted relationships between entities
    summary: str  # Generated summary of the story
    embedding: List[float]  # Embedding vector representing the story's content
    analysis: str  # Detailed analysis of the story
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())  # Timestamp of when the analysis was performed

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
            openai_client (Any): Asynchronous OpenAI client for interacting with LLMs.
            config_manager (ConfigManager): Manages configuration settings for the processor.
            loggers (Dict[str, logging.Logger]): Dictionary of loggers for different modules.
        """
        # Set up the primary logger for this processor
        self.logger = loggers.get("EnhancedEntityProcessor", logging.getLogger("EnhancedEntityProcessor"))
        self.openai_client = openai_client
        self.config_manager = config_manager
        self.loggers = loggers

        # Initialize LRU (Least Recently Used) caches with configurable maximum sizes
        cache_config = config_manager.get_cache_config()
        self.wikipedia_cache = LRUCache(maxsize=cache_config.get("wikipedia", {}).get("maxsize", 1000))
        self.wikidata_cache = LRUCache(maxsize=cache_config.get("wikidata", {}).get("maxsize", 1000))

        # Initialize Qdrant client for vector storage and retrieval
        qdrant_config = self.config_manager.get_qdrant_config()
        self.qdrant_client = QdrantClient(
            url=qdrant_config.get("url", "http://localhost:6333"),
            api_key=qdrant_config.get("api_key", None),
        )
        self.stories_collection_name = qdrant_config.get("stories_collection", {}).get("name", "stories")
        self.reports_collection_name = qdrant_config.get("reports_collection", {}).get("name", "reports")
        self.vector_size = int(qdrant_config.get("vector_size", 1536))  # Dimension of the embedding vectors

        # Initialize aiohttp configuration settings for HTTP requests
        aiohttp_config = self.config_manager.get_aiohttp_config()
        timeout_seconds = aiohttp_config.get("timeout", 60)
        max_connections = aiohttp_config.get("max_connections", 10)

        # Setup aiohttp connector with connection limits
        connector = aiohttp.TCPConnector(limit=max_connections)
        timeout = aiohttp.ClientTimeout(total=timeout_seconds)

        # Create an aiohttp ClientSession with the specified connector and timeout
        # Reuse the session for all HTTP requests to maximize connection pooling and reduce overhead
        self.session = aiohttp.ClientSession(connector=connector, timeout=timeout)

        # Setup rate limiting using aiolimiter to control the request rate
        rate_limit = aiohttp_config.get("rate_limit", 5)  # Maximum number of requests per second
        self.limiter = AsyncLimiter(max_rate=rate_limit, time_period=1)

        # Initialize retry strategy using tenacity for handling transient failures in API calls
        retry_config = config_manager.get_retry_config()
        self.retry_strategy = retry(
            reraise=True,
            stop=stop_after_attempt(retry_config.get("retries", 5)),  # Maximum number of retries
            wait=wait_exponential(
                multiplier=retry_config.get("base_delay", 0.5),  # Initial delay between retries
                min=retry_config.get("base_delay", 0.5),  # Minimum delay
                max=retry_config.get("max_delay", 10),  # Maximum delay
            ),
            retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError)),  # Exceptions to retry on
        )

        # Initialize TF-IDF Vectorizer for context matching in text analysis
        # Pre-initialize with stop words to reduce noise in similarity calculations
        self.vectorizer = TfidfVectorizer(stop_words="english")

        # Initialize ReportPostProcessor for post-processing generated reports
        self.report_post_processor = ReportPostProcessor(
            self.openai_client, self.config_manager, self.loggers
        )

        # Initialize a dictionary to store stories loaded for processing
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

        This method leverages a Large Language Model (LLM) to analyze the context in which an entity appears
        and suggests the most likely correct spelling if the entity name appears to be misspelled.

        Args:
            entity (str): The potentially misspelled entity name.
            context (str): Contextual information surrounding the entity.

        Returns:
            Optional[str]: The corrected entity name or None if unsuccessful.
        """
        # Construct the prompt for OpenAI to suggest correct spelling
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
            # Retrieve model configuration for chat models
            model_config = self.config_manager.get_model_config(ModelType.CHAT)
            # Make a request to OpenAI's chat completion API
            response = await self.openai_client.chat.completions.create(
                model=model_config["model"],
                messages=messages,
                max_tokens=10,  # Limit response length for efficiency
                temperature=0.0,  # Low temperature for deterministic output
            )
            # Check if the response contains valid choices
            if response.choices and response.choices[0].message:
                corrected = response.choices[0].message.content.strip()
                # Basic validation to ensure the correction is meaningful
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

        This method utilizes an LLM to propose alternative entity names that might be more accurate or commonly
        used, aiding in resolving ambiguities or inaccuracies in entity recognition.

        Args:
            entity (str): The original entity name.
            context (str): Contextual information surrounding the entity.

        Returns:
            Optional[str]: Suggested alternative name or None if unsuccessful.
        """
        # Retrieve system and user prompts from the configuration manager or use default prompts
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
            # Retrieve model configuration for chat models
            model_config = self.config_manager.get_model_config(ModelType.CHAT)
            # Make a request to OpenAI's chat completion API
            response = await self.openai_client.chat.completions.create(
                model=model_config["model"],
                messages=messages,
                max_tokens=50,  # Allow enough tokens for multiple suggestions
                temperature=model_config.get("temperature", 0.7),  # Moderate creativity
            )
            # Check if the response contains valid choices
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

        This method attempts to retrieve a unique identifier for an entity from Wikidata,
        enhancing accuracy through fuzzy and phonetic matching techniques when exact matches are not found.

        Args:
            entity (str): The entity name to search for.
            context (str): Contextual information surrounding the entity.

        Returns:
            Optional[str]: The Wikidata ID if found, else None.
        """
        # Check if the Wikidata ID is already cached to avoid redundant API calls
        if entity in self.wikidata_cache:
            self.loggers["wikipedia"].debug(f"Found Wikidata ID for '{entity}' in cache.")
            return self.wikidata_cache[entity]

        # Define the Wikidata API URL and parameters for searching entities
        wikidata_api_url = "https://www.wikidata.org/w/api.php"
        params = {
            "action": "wbsearchentities",
            "format": "json",
            "language": "en",
            "search": entity,
            "type": "item",
            "limit": 20,  # Limit the number of search results for efficiency
        }

        try:
            # Use the rate limiter to control the request rate
            async with self.limiter:
                # Make a GET request to the Wikidata API
                async with self.session.get(wikidata_api_url, params=params) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        search_results = data.get("search", [])
                        if search_results:
                            # If an exact match is found, return the Wikidata ID
                            wikidata_id = search_results[0].get("id")
                            self.wikidata_cache[entity] = wikidata_id  # Cache the ID for future use
                            self.loggers["wikipedia"].debug(f"Fetched Wikidata ID for '{entity}': {wikidata_id}")
                            return wikidata_id
            # If no exact match found, perform Fuzzy and Phonetic Matching
            return await self.perform_fuzzy_and_phonetic_matching(entity, context)
        except Exception as e:
            # Log any exceptions that occur during the API request
            self.loggers["wikipedia"].error(f"Error fetching Wikidata ID for '{entity}': {e}")
            return None

    @retry
    async def perform_fuzzy_and_phonetic_matching(self, entity: str, context: str) -> Optional[str]:
        """
        Perform Fuzzy and Phonetic matching to find the best Wikidata ID for the given entity.

        This method enhances entity resolution by matching entities that are similar in spelling
        (fuzzy matching) or pronunciation (phonetic matching) to account for variations and misspellings.

        Args:
            entity (str): The entity name to search for.
            context (str): Contextual information surrounding the entity.

        Returns:
            Optional[str]: The best matched Wikidata ID or None if not found.
        """
        # Define the Wikidata API URL and parameters for searching entities
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
            # Use the rate limiter to control the request rate
            async with self.limiter:
                # Make a GET request to the Wikidata API
                async with self.session.get(wikidata_api_url, params=params) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        search_results = data.get("search", [])
                        if not search_results:
                            self.loggers["wikipedia"].warning(f"No search results found for '{entity}' during fuzzy and phonetic matching.")
                            return None

                        # Prepare a list of entity names from search results
                        candidate_entities = [result["label"] for result in search_results]

                        # Fuzzy Matching using RapidFuzz to find the best approximate match
                        best_fuzzy_match, fuzzy_score, _ = process.extractOne(
                            query=entity, choices=candidate_entities, scorer=fuzz.WRatio
                        )
                        self.loggers["wikipedia"].debug(f"Best fuzzy match for '{entity}': '{best_fuzzy_match}' with score {fuzzy_score}")

                        # Define a similarity threshold for fuzzy matching to consider a match valid
                        matching_config = self.config_manager.config.get("matching", {})
                        fuzzy_threshold = matching_config.get("fuzzy_threshold", 80)

                        if fuzzy_score >= fuzzy_threshold:
                            # If the fuzzy match score exceeds the threshold, retrieve the corresponding Wikidata ID
                            for result in search_results:
                                if result["label"] == best_fuzzy_match:
                                    wikidata_id = result["id"]
                                    self.wikidata_cache[entity] = wikidata_id  # Cache the ID
                                    self.loggers["wikipedia"].debug(f"Fuzzy matched Wikidata ID for '{entity}': {wikidata_id}")
                                    return wikidata_id

                        # Phonetic Matching using Metaphone to find entities with similar pronunciation
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

                        # Additional Step: Contextual Validation to ensure relevance based on context
                        contextual_valid_id = await self.contextual_validate(entity, context, search_results)
                        if contextual_valid_id:
                            return contextual_valid_id

                        return None
            return None
        except Exception as e:
            # Log any exceptions that occur during the fuzzy and phonetic matching process
            self.loggers["wikipedia"].error(f"Error during fuzzy and phonetic matching for '{entity}': {e}")
            return None

    async def contextual_validate(
        self, entity: str, context: str, search_results: List[Dict[str, Any]]
    ) -> Optional[str]:
        """
        Validate entities based on context by comparing context embeddings with entity descriptions.

        This method uses TF-IDF vectorization and cosine similarity to measure how well an entity's description
        aligns with the surrounding context, ensuring that the entity is relevant to the specific usage.

        Args:
            entity (str): The original entity name.
            context (str): Contextual information surrounding the entity.
            search_results (List[Dict[str, Any]]): Search results from Wikidata API.

        Returns:
            Optional[str]: Validated Wikidata ID or None.
        """
        try:
            # Extract descriptions from search results for comparison
            descriptions = [result.get("description", "") for result in search_results]
            if not descriptions:
                self.loggers["wikipedia"].warning(f"No descriptions available for contextual validation of '{entity}'.")
                return None

            # Fit the vectorizer on descriptions and context to create vector representations
            # To optimize, fit once and reuse if possible, but since context varies, we fit per call
            self.vectorizer.fit(descriptions + [context])

            # Transform context and descriptions into vector space
            context_vector = self.vectorizer.transform([context])
            description_vectors = self.vectorizer.transform(descriptions)

            # Calculate cosine similarity between context and each description
            similarities = cosine_similarity(context_vector, description_vectors).flatten()
            self.loggers["wikipedia"].debug(f"Cosine similarities for '{entity}': {similarities}")

            # Identify the index with the highest similarity score
            best_match_index = similarities.argmax()
            best_similarity = similarities[best_match_index]
            self.loggers["wikipedia"].debug(f"Best contextual similarity for '{entity}': {best_similarity} at index {best_match_index}")

            # Define a similarity threshold for contextual validation
            matching_config = self.config_manager.config.get("matching", {})
            context_threshold = matching_config.get("context_threshold", 0.3)

            if best_similarity >= context_threshold:
                # If similarity exceeds threshold, retrieve the corresponding Wikidata ID
                best_result = search_results[best_match_index]
                wikidata_id = best_result.get("id")
                self.wikidata_cache[entity] = wikidata_id  # Cache the ID
                self.loggers["wikipedia"].debug(f"Contextually validated Wikidata ID for '{entity}': {wikidata_id}")
                return wikidata_id
            else:
                self.loggers["wikipedia"].warning(f"Contextual similarity below threshold for '{entity}'.")
                return None
        except Exception as e:
            # Log any exceptions that occur during contextual validation
            self.loggers["wikipedia"].error(f"Error during contextual validation for '{entity}': {e}")
            raise ProcessingError(f"Contextual validation failed: {e}")

    @retry
    async def validate_entity(self, wikidata_id: str, context: str) -> bool:
        """
        Validate the entity by checking essential properties on Wikidata and ensuring alignment with context.

        This method ensures that the entity retrieved from Wikidata possesses necessary attributes and that
        its description aligns with the context in which it appears, enhancing the accuracy of entity recognition.

        Args:
            wikidata_id (str): The Wikidata ID of the entity.
            context (str): Contextual information surrounding the entity.

        Returns:
            bool: True if the entity is valid and aligns with context, False otherwise.
        """
        # Construct the Wikidata API URL for fetching entity data
        wikidata_api_url = f"https://www.wikidata.org/wiki/Special:EntityData/{wikidata_id}.json"
        try:
            # Use the rate limiter to control the request rate
            async with self.limiter:
                # Make a GET request to the Wikidata API
                async with self.session.get(wikidata_api_url) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        entity_data = data.get("entities", {}).get(wikidata_id, {})
                        claims = entity_data.get("claims", {})
                        descriptions = entity_data.get("descriptions", {})
                        english_description = descriptions.get("en", {}).get("value", "")
                        # Example validation: Check if the entity has an occupation (P106)
                        if "P106" in claims:
                            # Further validate alignment with context based on keywords
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
            # Log any exceptions that occur during the validation process
            self.loggers["wikipedia"].error(f"Error validating entity '{wikidata_id}': {e}")
            return False

    @retry
    async def get_entity_sections(self, title: str) -> Dict[str, Any]:
        """
        Fetch specific sections from a Wikipedia page.

        This method retrieves structured sections from a Wikipedia page, allowing for targeted summarization
        of relevant content areas.

        Args:
            title (str): The title of the Wikipedia page.

        Returns:
            Dict[str, Any]: A dictionary of section titles and their corresponding summarized text.
        """
        # Define the Wikipedia API URL for fetching mobile sections
        sections_api_url = f"https://en.wikipedia.org/api/rest_v1/page/mobile-sections/{title}"
        try:
            # Use the rate limiter to control the request rate
            async with self.limiter:
                # Make a GET request to the Wikipedia API
                async with self.session.get(sections_api_url) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        sections = data.get("sections", [])
                        section_contents = {}
                        for section in sections:
                            section_title = section.get("title", "No Title")
                            section_text = section.get("text", "")
                            if section_text:
                                # Summarize the text of each section using an LLM
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

        Leveraging an LLM, this method generates concise summaries of larger text sections,
        facilitating easier consumption and analysis of information.

        Args:
            text (str): The text to be summarized.

        Returns:
            str: The summarized text.
        """
        # Construct the prompt for text summarization
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
            # Retrieve model configuration for chat models
            model_config = self.config_manager.get_model_config(ModelType.CHAT)
            # Make a request to OpenAI's chat completion API for summarization
            response = await self.openai_client.chat.completions.create(
                model=model_config["model"],
                messages=messages,
                max_tokens=150,  # Limit response length for concise summaries
                temperature=0.5,  # Moderate creativity for balanced summaries
            )
            # Check if the response contains valid choices
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
            # Log any exceptions that occur during summarization
            self.loggers["llm"].error(f"Failed to summarize text: {e}")
            return text  # Fallback to original text in case of exception

    @retry_async()
    async def _get_embedding_async(self, text: str) -> List[float]:
        """
        Generate an embedding for the given text using OpenAI's embedding model.

        Embeddings are numerical representations of text that capture semantic meaning,
        enabling tasks like similarity comparison and clustering in vector databases.

        Args:
            text (str): The text to generate an embedding for.

        Returns:
            List[float]: The generated embedding vector.
        """
        try:
            # Retrieve model configuration for embedding models
            model_config = self.config_manager.get_model_config(ModelType.EMBEDDING)
            # Make a request to OpenAI's embeddings API
            response = await self.openai_client.embeddings.create(
                input=[text],
                model=model_config["model"],
            )
            # Validate the response contains embedding data
            if not response.data or not response.data[0].embedding:
                self.loggers["llm"].error("OpenAI response is missing embedding data.")
                raise ProcessingError("Failed to generate embedding due to invalid OpenAI response.")
            embedding = response.data[0].embedding
            self.loggers["llm"].debug(
                f"Generated embedding for text: {text[:30]}... [truncated]. Embedding size: {len(embedding)}"
            )
            return embedding
        except Exception as e:
            # Log any exceptions that occur during embedding generation
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

        This method structures the data in a format suitable for storage in Qdrant,
        including the embedding vector and associated metadata.

        Args:
            collection_name (str): Name of the Qdrant collection.
            point_id (str): Unique identifier for the point.
            content (str): The content to be stored.
            wiki_info (Dict[str, Any]): Wikipedia information associated with the content.
            embedding (List[float]): Embedding vector.
            story_id (str): Identifier for the story.

        Returns:
            List[models.PointStruct]: List containing the prepared point.
        """
        # Validate the embedding size to match Qdrant's vector requirements
        if not isinstance(embedding, list) or len(embedding) != self.vector_size:
            raise ValueError(f"Invalid embedding: expected list of length {self.vector_size}")
        # Construct the payload with relevant metadata
        payload = {
            "report_id": point_id,
            "story_id": story_id,
            "content": content,
            "wiki_info": json.dumps(wiki_info),
            "timestamp": datetime.now().isoformat(),
        }
        # Create a PointStruct object representing the data point
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

        UUID4 generates a random UUID, ensuring uniqueness across reports.

        Returns:
            str: The generated UUID4 string.
        """
        return str(uuid.uuid4())

    def _store_points_in_qdrant_sync(self, collection_name: str, points: List[models.PointStruct]) -> None:
        """
        Synchronously store points in Qdrant.

        This method performs the actual upsert operation to store vectors and metadata in Qdrant.

        Args:
            collection_name (str): Name of the Qdrant collection.
            points (List[models.PointStruct]): List of points to store.

        Raises:
            ProcessingError: If storing points fails.
        """
        try:
            # Perform the upsert operation to store points in the specified collection
            # Using bulk upsert for efficiency
            operation_info = self.qdrant_client.upsert(collection_name=collection_name, points=points)
            self.loggers["main"].info(
                f"Stored {len(points)} point(s) in Qdrant collection '{collection_name}'. Operation ID: {operation_info.operation_id}"
            )
            self.loggers["main"].info(f"Upsert operation completed for collection '{collection_name}'")
        except UnexpectedResponse as e:
            # Raise a custom ProcessingError if Qdrant responds unexpectedly
            raise ProcessingError(f"Failed to store points in Qdrant: {e}")
        except Exception as e:
            # Raise a custom ProcessingError for any other unexpected errors
            raise ProcessingError(f"An unexpected error occurred while storing points in Qdrant: {e}")

    async def _store_points_in_qdrant(self, collection_name: str, points: List[models.PointStruct]) -> None:
        """
        Asynchronously store points in Qdrant by running the synchronous method in an executor.

        This method allows the synchronous upsert operation to be performed without blocking the event loop.

        Args:
            collection_name (str): Name of the Qdrant collection.
            points (List[models.PointStruct]): List of points to store.
        """
        # Get the current event loop
        loop = asyncio.get_running_loop()
        # Run the synchronous storage method in a separate thread to avoid blocking
        await loop.run_in_executor(None, self._store_points_in_qdrant_sync, collection_name, points)

    async def perform_sentiment_analysis(self, text: str) -> str:
        """
        Perform sentiment analysis on the given text.

        This method uses an LLM to determine the overall sentiment expressed in the text,
        categorizing it as 'positive', 'negative', or 'neutral'.

        Args:
            text (str): The text to analyze.

        Returns:
            str: The sentiment result ('positive', 'negative', or 'neutral').
        """
        # Construct the prompt for sentiment analysis
        prompt = f"Analyze the sentiment of the following text and respond with 'positive', 'negative', or 'neutral':\n\n{text}"
        messages = [{"role": "user", "content": prompt}]
        try:
            # Retrieve model configuration for chat models
            model_config = self.config_manager.get_model_config(ModelType.CHAT)
            # Make a request to OpenAI's chat completion API for sentiment analysis
            response = await self.openai_client.chat.completions.create(
                model=model_config["model"],
                messages=messages,
                max_tokens=1,  # Short response expected
                temperature=0.0,  # Deterministic output
            )
            # Extract and process the sentiment result
            sentiment = response.choices[0].message.content.strip().lower()
            if sentiment in ["positive", "negative", "neutral"]:
                return sentiment
            else:
                self.loggers["llm"].warning(f"Unexpected sentiment result: {sentiment}")
                return "unknown"
        except Exception as e:
            # Log any exceptions that occur during sentiment analysis
            self.loggers["errors"].error(f"Sentiment analysis failed: {e}")
            return "unknown"

    async def extract_concepts(self, text: str) -> List[str]:
        """
        Extract key concepts from the text.

        Utilizing an LLM, this method identifies significant concepts within the text,
        aiding in the understanding and categorization of the content.

        Args:
            text (str): The text to analyze.

        Returns:
            List[str]: A list of extracted concepts.
        """
        # Construct the prompt for concept extraction
        prompt = f"Identify the key concepts in the following text:\n\n{text}\n\nConcepts:"
        messages = [{"role": "user", "content": prompt}]
        try:
            # Retrieve model configuration for chat models
            model_config = self.config_manager.get_model_config(ModelType.CHAT)
            # Make a request to OpenAI's chat completion API for concept extraction
            response = await self.openai_client.chat.completions.create(
                model=model_config["model"],
                messages=messages,
                max_tokens=50,  # Allow enough tokens for multiple concepts
                temperature=0.5,  # Moderate creativity
            )
            # Extract and process the concepts from the response
            concepts = response.choices[0].message.content.strip()
            concepts_list = [concept.strip() for concept in concepts.split(",") if concept.strip()]
            return concepts_list
        except Exception as e:
            # Log any exceptions that occur during concept extraction
            self.loggers["errors"].error(f"Concept extraction failed: {e}")
            return []

    async def perform_entity_sentiment_analysis(self, entity: str, context: str) -> str:
        """
        Analyze the sentiment associated with a specific entity within the context.

        This method focuses on determining the sentiment directed towards a particular entity,
        providing more granular sentiment insights.

        Args:
            entity (str): The entity to analyze.
            context (str): The full text context.

        Returns:
            str: The sentiment result ('positive', 'negative', or 'neutral').
        """
        # Construct the prompt for entity-specific sentiment analysis
        prompt = f"In the following text, what is the sentiment towards '{entity}'? Respond with 'positive', 'negative', or 'neutral':\n\n{context}"
        messages = [{"role": "user", "content": prompt}]
        try:
            # Retrieve model configuration for chat models
            model_config = self.config_manager.get_model_config(ModelType.CHAT)
            # Make a request to OpenAI's chat completion API for entity sentiment analysis
            response = await self.openai_client.chat.completions.create(
                model=model_config["model"],
                messages=messages,
                max_tokens=1,  # Short response expected
                temperature=0.0,  # Deterministic output
            )
            # Extract and process the sentiment result
            sentiment = response.choices[0].message.content.strip().lower()
            if sentiment in ["positive", "negative", "neutral"]:
                return sentiment
            else:
                self.loggers["llm"].warning(f"Unexpected entity sentiment result for '{entity}': {sentiment}")
                return "unknown"
        except Exception as e:
            # Log any exceptions that occur during entity sentiment analysis
            self.loggers["errors"].error(f"Entity sentiment analysis failed for '{entity}': {e}")
            return "unknown"

    async def extract_entity_relations(self, entities: List[str], text: str) -> Any:
        """
        Extract relationships between entities in the text.

        Leveraging an LLM, this method identifies and articulates the relationships between different entities
        mentioned within the text, contributing to a richer understanding of the content's structure and connections.

        Args:
            entities (List[str]): A list of entity names.
            text (str): The text to analyze.

        Returns:
            Any: The extracted relationships.
        """
        # Construct the prompt for relationship extraction
        prompt = f"Extract relationships between the following entities in the text: {', '.join(entities)}.\n\nText:\n{text}\n\nList the relationships:"
        messages = [{"role": "user", "content": prompt}]
        try:
            # Retrieve model configuration for chat models
            model_config = self.config_manager.get_model_config(ModelType.CHAT)
            # Make a request to OpenAI's chat completion API for relationship extraction
            response = await self.openai_client.chat.completions.create(
                model=model_config["model"],
                messages=messages,
                max_tokens=150,  # Allow enough tokens for detailed relationships
                temperature=0.5,  # Moderate creativity
            )
            # Extract and return the relationships from the response
            relations_text = response.choices[0].message.content.strip()
            return relations_text
        except Exception as e:
            # Log any exceptions that occur during relationship extraction
            self.loggers["errors"].error(f"Relation extraction failed: {e}")
            return []

    async def perform_emotion_analysis(self, text: str) -> List[str]:
        """
        Analyze the emotions expressed in the text.

        This method uses an LLM to identify and list the emotions conveyed within the text,
        providing deeper emotional insights into the content.

        Args:
            text (str): The text to analyze.

        Returns:
            List[str]: A list of detected emotions.
        """
        # Construct the prompt for emotion analysis
        prompt = f"Identify the emotions expressed in the following text:\n\n{text}\n\nEmotions:"
        messages = [{"role": "user", "content": prompt}]
        try:
            # Retrieve model configuration for chat models
            model_config = self.config_manager.get_model_config(ModelType.CHAT)
            # Make a request to OpenAI's chat completion API for emotion analysis
            response = await self.openai_client.chat.completions.create(
                model=model_config["model"],
                messages=messages,
                max_tokens=50,  # Allow enough tokens for multiple emotions
                temperature=0.5,  # Moderate creativity
            )
            # Extract and process the emotions from the response
            emotions = response.choices[0].message.content.strip()
            emotions_list = [emotion.strip() for emotion in emotions.split(",") if emotion.strip()]
            return emotions_list
        except Exception as e:
            # Log any exceptions that occur during emotion analysis
            self.loggers["errors"].error(f"Emotion analysis failed: {e}")
            return []

    def extract_keywords(self, text: str) -> List[str]:
        """
        Extract keywords from the text.

        Utilizing YAKE (Yet Another Keyword Extractor), this method identifies prominent keywords
        that capture the essence of the text, aiding in indexing and search functionalities.

        Args:
            text (str): The text to analyze.

        Returns:
            List[str]: A list of extracted keywords.
        """
        try:
            # Initialize the YAKE keyword extractor with specified parameters
            kw_extractor = yake.KeywordExtractor(lan="en", n=1, top=10)
            # Extract keywords from the text
            keywords = kw_extractor.extract_keywords(text)
            # Process and return the list of keywords
            keywords_list = [kw[0] for kw in keywords]
            return keywords_list
        except Exception as e:
            # Log any exceptions that occur during keyword extraction
            self.loggers["errors"].error(f"Keyword extraction failed: {e}")
            return []

    def detect_language(self, text: str) -> str:
        """
        Detect the language of the given text.

        This method uses the `langdetect` library to identify the language in which the text is written,
        facilitating appropriate preprocessing steps like translation if necessary.

        Args:
            text (str): The text to analyze.

        Returns:
            str: The ISO 639-1 code of the detected language.
        """
        try:
            # Use langdetect to identify the language of the text
            language = detect(text)
            return language
        except Exception as e:
            # Log any exceptions that occur during language detection
            self.loggers["errors"].error(f"Language detection failed: {e}")
            return "unknown"

    async def translate_text(self, text: str, target_lang: str = "en") -> str:
        """
        Translate the given text to the target language.

        This method leverages an LLM to perform translation of text from its detected language
        to a specified target language, enhancing the processor's ability to handle multilingual content.

        Args:
            text (str): The text to translate.
            target_lang (str): The target language code (default is English).

        Returns:
            str: The translated text.
        """
        # Construct the prompt for translation
        prompt = f"Translate the following text to {target_lang}:\n\n{text}"
        messages = [{"role": "user", "content": prompt}]
        try:
            # Retrieve model configuration for chat models
            model_config = self.config_manager.get_model_config(ModelType.CHAT)
            # Make a request to OpenAI's chat completion API for translation
            response = await self.openai_client.chat.completions.create(
                model=model_config["model"],
                messages=messages,
                max_tokens=1000,  # Allow enough tokens for long translations
                temperature=0.3,  # Low temperature for accuracy
            )
            # Extract and return the translated text from the response
            translated_text = response.choices[0].message.content.strip()
            self.loggers["llm"].debug(f"Translated text: {translated_text[:30]}... [truncated]")
            return translated_text
        except Exception as e:
            # Log any exceptions that occur during translation
            self.loggers["main"].error(f"Translation failed: {e}")
            return text  # Fallback to original text if translation fails

    def generate_knowledge_graph(self, entities: List[Dict[str, str]], relations: Any) -> None:
        """
        Generate a knowledge graph from entities and their relationships.

        This method constructs a knowledge graph that visually and structurally represents the
        entities extracted from the text and the relationships identified between them,
        facilitating enhanced data analysis and visualization.

        Args:
            entities (List[Dict[str, str]]): List of entity dictionaries.
            relations (Any): The relationships between entities.
        """
        try:
            # Placeholder: Actual implementation would involve creating nodes and edges
            # based on entities and their relationships, possibly using a graph database or visualization tool
            # For demonstration, we'll log the entities and relations
            self.loggers["main"].info("Generating knowledge graph from entities and relationships.")
            for entity in entities:
                self.loggers["main"].debug(f"Entity: {entity['text']}")
            self.loggers["main"].debug(f"Relations: {relations}")
            # TODO: Implement actual knowledge graph generation logic
            self.loggers["main"].info("Knowledge graph generation completed successfully.")
        except Exception as e:
            # Log any exceptions that occur during knowledge graph generation
            self.loggers["errors"].error(f"Knowledge graph generation failed: {e}")

    @retry_async()
    async def process_story_async(self, story_id: str, content: str) -> ProcessingResult:
        """
        Process a single story asynchronously, including preprocessing, entity resolution,
        embedding generation, and storing in Qdrant.

        This comprehensive method orchestrates the entire processing pipeline for a story,
        integrating various AI-driven analyses to extract, analyze, and store meaningful information.

        Args:
            story_id (str): Unique identifier for the story.
            content (str): The content of the story.

        Returns:
            ProcessingResult: Result of the processing, indicating success or failure.
        """
        try:
            # Detect the language of the story content
            language = self.detect_language(content)
            if language != 'en':
                # Translate the content to English if it's not already
                corrected_content = await self.translate_text(content)
            else:
                corrected_content = content

            self.loggers["main"].debug(f"Starting preprocessing for story '{story_id}'")
            # Preprocess the text using an external worker function (e.g., cleaning, normalization)
            # Offloading CPU-bound tasks to executor to prevent blocking the event loop
            corrected_content = await asyncio.get_running_loop().run_in_executor(
                None, preprocess_text_worker, corrected_content
            )
            self.loggers["main"].debug(f"Preprocessing completed for story '{story_id}'")

            # Extract and resolve entities using an external worker function (e.g., NER)
            # Offloading CPU-bound tasks to executor to prevent blocking the event loop
            resolved_entities = await asyncio.get_running_loop().run_in_executor(
                None, extract_and_resolve_entities_worker, corrected_content
            )
            self.loggers["main"].debug(f"Entities extracted for story '{story_id}': {resolved_entities}")

            # Retrieve comprehensive information about the extracted entities
            wiki_info = await self.get_entities_info(resolved_entities, corrected_content)

            # Perform various analyses on the corrected content
            sentiment = await self.perform_sentiment_analysis(corrected_content)
            concepts = await self.extract_concepts(corrected_content)
            emotions = await self.perform_emotion_analysis(corrected_content)
            keywords = self.extract_keywords(corrected_content)
            relations = await self.extract_entity_relations(
                [entity['text'] for entity in resolved_entities], corrected_content
            )

            # Generate analysis using the ReportPostProcessor, which may involve further LLM interactions
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
            # Save intermediary analysis data asynchronously
            await self._save_intermediary_data_async(
                data=analysis,
                path=f"analysis/{story_id}.json",
                data_type="analysis",
            )

            # Create an instance of StoryAnalysisResult to encapsulate all analysis data
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

            # Generate embedding for the corrected content
            self.loggers["llm"].debug(f"Generating embedding for story '{story_id}'")
            embedding = await self._get_embedding_async(corrected_content)
            self.loggers["llm"].debug(f"Embedding generated for story '{story_id}'")
            # Save the embedding data asynchronously
            await self._save_intermediary_data_async(
                data=embedding,
                path=f"embeddings/{story_id}.json",
                data_type="embedding",
            )

            # Store the corrected content and embedding in the Qdrant stories collection
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

            # Store the refined analysis in the Qdrant reports collection
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
            # Log any processing-specific errors
            self.loggers["errors"].error(f"Failed to process story '{story_id}': {pe}")
            return ProcessingResult(success=False, error=str(pe))
        except Exception as e:
            # Log any unexpected exceptions
            self.loggers["errors"].error(f"Failed to process story '{story_id}': {e}")
            return ProcessingResult(success=False, error=str(e))

    async def process_stories_async(
        self, stories_dir: str, output_path: str, summary_output_path: str
    ) -> None:
        """
        Process multiple stories asynchronously, handling concurrency, and generating reports.

        This method orchestrates the processing of multiple stories concurrently,
        managing resources efficiently and generating comprehensive and summary reports.

        Args:
            stories_dir (str): Directory containing story files.
            output_path (str): Path to save the refined report.
            summary_output_path (str): Path to save the summary report.
        """
        try:
            # Load stories from the specified directory
            self._load_stories(stories_dir)
            # Retrieve concurrency configuration
            concurrency_config = self.config_manager.get_concurrency_config()
            semaphore_limit = concurrency_config.get(
                "semaphore_limit", asyncio.cpu_count() * 2
            )
            # Initialize a semaphore to control the number of concurrent tasks
            semaphore = asyncio.Semaphore(semaphore_limit)

            async def semaphore_wrapper(story_id, content):
                """
                Wrapper function to process stories within the semaphore limit.

                Args:
                    story_id (str): Unique identifier for the story.
                    content (str): The content of the story.

                Returns:
                    ProcessingResult: Result of processing the story.
                """
                async with semaphore:
                    return await self.process_story_async(story_id, content)

            tasks = []
            # Create asynchronous tasks for processing each story
            for story_id, content in self.stories.items():
                self.loggers["main"].info(f"Processing story: {story_id}")
                tasks.append(semaphore_wrapper(story_id, content))
            # Execute all tasks concurrently with limited concurrency
            # Using asyncio.gather ensures that all tasks are awaited and results are collected
            results = await asyncio.gather(*tasks, return_exceptions=True)
            processed_results = []
            wiki_info = {}
            # Process the results of each task
            for result in results:
                if isinstance(result, ProcessingResult):
                    processed_results.append(result)
                    if result.success and result.data:
                        wiki_info[result.data.story_id] = result.data.wiki_info
                elif isinstance(result, Exception):
                    # Log any exceptions that occurred during processing
                    self.loggers["errors"].error(f"Error processing story: {result}")
                    processed_results.append(
                        ProcessingResult(success=False, error=str(result))
                    )
                else:
                    # Handle unexpected result types
                    self.loggers["errors"].error(
                        f"Unexpected result type: {type(result)}"
                    )
                    processed_results.append(
                        ProcessingResult(success=False, error="Unexpected result type")
                    )

            # Generate a refined report from all processed stories using the ReportPostProcessor
            refined_report = await self.report_post_processor.process_full_report(
                processed_results
            )
            # Save the refined report to the specified output path
            await self._save_report_async(refined_report, output_path)

            # Store refined reports in the Qdrant reports collection
            # To optimize performance, batch embeddings and upserts if possible
            # Here, we process each report individually to maintain association
            # Further optimization could involve collecting multiple reports before upserting
            for result in processed_results:
                if result.success and result.data:
                    story_id = result.data.story_id
                    report_content = result.data.analysis
                    wiki_info_content = json.dumps(result.data.wiki_info, indent=2)
                    report_id = self._generate_report_id()
                    # Generate embedding for the report content
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

            # Generate a summary report from all stored reports
            await self.generate_summary_report(summary_output_path)

            self.loggers["main"].info(
                f"Processing complete. Refined reports saved to '{self.reports_collection_name}' and summary saved to {summary_output_path}"
            )
        except Exception as e:
            # Log any exceptions that occur during the processing of stories
            self.loggers["errors"].error(
                f"Failed to process stories: {e}", exc_info=True
            )
        finally:
            # Ensure that the aiohttp session is closed
            if self.session:
                try:
                    await self.close()
                except Exception as e:
                    self.loggers["errors"].error(
                        f"Failed to close aiohttp session: {e}"
                    )
            # Ensure that the Qdrant client is properly closed
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

        This method scans the given directory for Markdown files, reads their content asynchronously,
        and stores them in a dictionary for processing.

        Args:
            stories_dir (str): Directory containing story files.

        Raises:
            ProcessingError: If no valid stories are found.
        """
        # Define the path to the stories directory
        stories_path = Path(stories_dir)
        # Check if the directory exists and is indeed a directory
        if not stories_path.exists() or not stories_path.is_dir():
            raise ProcessingError(f"Stories directory not found: {stories_dir}")
        self.stories = {}
        # Iterate over all Markdown files in the directory
        for story_file in stories_path.glob("*.md"):
            try:
                async def read_file():
                    """
                    Asynchronous function to read the content of a story file.
                    """
                    async with aiofiles.open(
                        story_file, "r", encoding="utf-8", errors="replace"
                    ) as f:
                        return await f.read()

                # Read the content of the story file
                content = asyncio.run(read_file()).strip()
                if not content:
                    # Log a warning if the story file is empty
                    self.loggers["wikipedia"].warning(
                        f"Story file '{story_file}' is empty. Skipping."
                    )
                    continue
                # Store the story content with the filename (without extension) as the ID
                self.stories[story_file.stem] = content
                self.loggers["main"].debug(f"Loaded story '{story_file.stem}'")
            except Exception as e:
                # Log any exceptions that occur while reading a story file
                self.loggers["errors"].error(
                    f"Failed to read story file '{story_file}': {e}"
                )
                continue
        # Check if any stories were successfully loaded
        if not self.stories:
            raise ProcessingError("No valid stories found to process.")
        self.loggers["main"].info(
            f"Loaded {len(self.stories)} story/stories for processing."
        )

    async def _save_report_async(self, report_content: str, output_path: str) -> None:
        """
        Save the report content to the specified output path asynchronously.

        This method writes the generated report to a file, ensuring that the directory structure exists.

        Args:
            report_content (str): The content of the report.
            output_path (str): Path to save the report.
        """
        # Define the path to the output file
        output_file = Path(output_path)
        # Create the parent directories if they don't exist
        output_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            # Asynchronously write the report content to the file
            async with aiofiles.open(
                output_file, "w", encoding="utf-8", errors="replace"
            ) as f:
                await f.write(report_content)
            self.loggers["main"].info(f"Report saved to {output_path}")
        except Exception as e:
            # Log any exceptions that occur while saving the report
            self.loggers["errors"].error(
                f"Failed to save report to '{output_path}': {e}"
            )

    async def _save_intermediary_data_async(
        self, data: Any, path: str, data_type: str
    ) -> None:
        """
        Save intermediary data (e.g., embeddings, analysis) to a specified path asynchronously.

        This method stores various forms of processed data that may be useful for debugging,
        auditing, or further processing.

        Args:
            data (Any): The data to save.
            path (str): Relative path to save the data.
            data_type (str): Type of data being saved (for logging purposes).
        """
        # Retrieve the intermediary directory from configuration
        intermediary_dir = self.config_manager.get_paths_config().get(
            "intermediary_dir", "output/intermediary"
        )
        # Define the full path to save the data
        output_dir = Path(intermediary_dir) / path
        # Create the parent directories if they don't exist
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        try:
            if data_type in [
                "entities",
                "wiki_info",
                "embedding",
                "analysis",
                "additional_info",
            ]:
                # Asynchronously write JSON data to the file
                async with aiofiles.open(
                    output_dir, "w", encoding="utf-8", errors="replace"
                ) as f:
                    await f.write(json.dumps(data, indent=2))
                self.loggers["main"].debug(f"Saved {data_type} for '{path}'")
            elif data_type == "preprocessed_text":
                # Asynchronously write plain text data to the file
                async with aiofiles.open(
                    output_dir, "w", encoding="utf-8", errors="replace"
                ) as f:
                    await f.write(data)
                self.loggers["main"].debug(f"Saved preprocessed text for '{path}'")
            else:
                # Log a warning for unknown data types
                self.loggers["main"].warning(
                    f"Unknown data_type '{data_type}' for saving intermediary data."
                )
        except Exception as e:
            # Log any exceptions that occur while saving intermediary data
            self.loggers["errors"].error(
                f"Failed to save intermediary data to '{path}': {e}"
            )

    def _retrieve_all_reports_sync(self, collection_name: str) -> List[str]:
        """
        Synchronously retrieve all reports from a specified Qdrant collection.

        This method fetches all stored reports from a Qdrant collection,
        facilitating the generation of summary reports or further analysis.

        Args:
            collection_name (str): Name of the Qdrant collection.

        Returns:
            List[str]: List of report contents.
        """
        try:
            all_reports = []
            # Retrieve the limit for how many reports to fetch from configuration
            limit = self.config_manager.get_qdrant_config().get("retrieve_limit", 1000)
            # Perform a scroll operation to retrieve all reports up to the specified limit
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
            # Iterate over each record and extract the report content
            for record in scroll_result:
                if isinstance(record, models.Record) and record.payload:
                    content = record.payload.get("content", "")
                    if content:
                        all_reports.append(content)
                else:
                    # Log a warning for unexpected record types or missing payloads
                    self.loggers["errors"].warning(
                        f"Unexpected record type or missing payload: {record}"
                    )
            self.loggers["main"].info(
                f"Retrieved {len(all_reports)} reports from '{collection_name}' collection."
            )
            return all_reports
        except Exception as e:
            # Log any exceptions that occur during retrieval
            self.loggers["errors"].error(
                f"Failed to retrieve reports from '{collection_name}' collection: {e}"
            )
            return []

    async def _retrieve_all_reports_async(self, collection_name: str) -> List[str]:
        """
        Asynchronously retrieve all reports from a specified Qdrant collection.

        This method wraps the synchronous retrieval method, allowing it to be called
        within an asynchronous context without blocking the event loop.

        Args:
            collection_name (str): Name of the Qdrant collection.

        Returns:
            List[str]: List of report contents.
        """
        # Get the current event loop
        loop = asyncio.get_running_loop()
        # Run the synchronous retrieval method in a separate thread to avoid blocking
        return await loop.run_in_executor(
            None, self._retrieve_all_reports_sync, collection_name
        )

    async def generate_summary_report(self, summary_output_path: str) -> None:
        """
        Generate a summary report from all stored reports and save it asynchronously.

        This method aggregates all individual reports, synthesizes a comprehensive summary
        using an LLM, and saves the summary to the specified path.

        Args:
            summary_output_path (str): Path to save the summary report.
        """
        # Retrieve all reports from the Qdrant reports collection
        all_reports = await self._retrieve_all_reports_async(
            self.reports_collection_name
        )
        if not all_reports:
            # Handle the case where no reports are available
            self.loggers["main"].warning(
                "No reports found in the reports collection to generate a summary."
            )
            summary = "No reports available to generate a summary."
        else:
            self.loggers["llm"].debug(
                "Generating summary from all stored reports"
            )
            # Generate a summary using the ReportPostProcessor, which utilizes an LLM
            summary = await self.report_post_processor.generate_summary(all_reports)
        # Save the generated summary to the specified output path
        await self._save_report_async(summary, summary_output_path)

    async def get_entities_info(
        self, entities: List[Dict[str, str]], context: str
    ) -> Dict[str, Any]:
        """
        Retrieve comprehensive information about a list of entities.

        This method gathers detailed information about each entity, including Wikipedia summaries,
        descriptions, and other relevant metadata, enhancing the depth of entity analysis.

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
                # Create a task to fetch information for each entity
                tasks.append(self.get_entity_info_single(entity, context))
            else:
                # Log an error if the entity dictionary is missing the 'text' field
                self.loggers['wikipedia'].error(f"Entity dictionary missing 'text' field: {entity_dict}")
        # Execute all entity information retrieval tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for entity, result in zip([e.get('text') for e in entities if 'text' in e], results):
            if isinstance(result, dict):
                # Perform entity-level sentiment analysis and add it to the entity info
                entity_sentiment = await self.perform_entity_sentiment_analysis(entity, context)
                result['sentiment'] = entity_sentiment
                wiki_info[entity] = result
            elif isinstance(result, Exception):
                # Log any exceptions that occurred during entity information retrieval
                self.loggers['wikipedia'].error(f"Error fetching info for entity '{entity}': {result}")
            else:
                # Handle cases where no information was found for the entity
                self.loggers['wikipedia'].warning(f"No information found for entity '{entity}'.")
        return wiki_info

    @retry
    async def get_entity_info_single(
        self, entity: str, context: str
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve comprehensive information about a single entity.

        This method performs spell correction, fetches Wikidata information, validates the entity,
        and retrieves detailed Wikipedia summaries and relevant sections.

        Args:
            entity (str): The entity name to retrieve information for.
            context (str): Contextual information for processing.

        Returns:
            Optional[Dict[str, Any]]: Dictionary containing information about the entity or None if not found.
        """
        try:
            # Step 1: Contextual Spell Correction using LLM
            corrected_entity = await self.correct_entity_spelling(entity, context)
            if corrected_entity and corrected_entity != entity:
                self.loggers["wikipedia"].info(
                    f"Corrected entity name from '{entity}' to '{corrected_entity}'"
                )
                entity = corrected_entity

            # Check if the entity information is already cached to avoid redundant API calls
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

            # Validate the entity using its Wikidata ID to ensure relevance and completeness
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
                            # Further validate alignment with context based on keywords
                            context_keywords = set(context.lower().split())
                            description_keywords = set(english_description.lower().split())
                            common_keywords = context_keywords.intersection(description_keywords)
                            if common_keywords:
                                self.loggers["wikipedia"].debug(
                                    f"Entity '{wikidata_id}' aligns with context based on keywords: {common_keywords}"
                                )
                                # Continue processing as the entity is relevant
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

                        # Retrieve site links to get the corresponding Wikipedia title
                        sitelinks = entity_data.get("sitelinks", {})
                        enwiki = sitelinks.get("enwiki", {})
                        wikipedia_title = enwiki.get("title")
                        if not wikipedia_title:
                            self.loggers["wikipedia"].warning(
                                f"No English Wikipedia page found for '{entity}'."
                            )
                            return None

                        # Fetch Wikipedia summary using the summary API
                        summary_api_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{wikipedia_title}"
                        async with self.limiter:
                            async with self.session.get(summary_api_url) as summary_resp:
                                if summary_resp.status == 200:
                                    summary_data = await summary_resp.json()
                                    if summary_data.get("type") == "disambiguation":
                                        # Handle disambiguation pages by suggesting alternative entities
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
                                    aliases = list(set(aliases))  # Remove duplicates

                                    categories = []
                                    if "P910" in claims:  # Additional name property (P910)
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
                        # Handle case where the Wikidata entity is not found
                        self.loggers["wikipedia"].warning(
                            f"Wikidata entity not found for '{entity}'."
                        )
                        return None
                    else:
                        # Log an error if the Wikidata entity request fails with a different status code
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

    async def _store_points_in_qdrant(self, collection_name: str, points: List[models.PointStruct]) -> None:
        """
        Asynchronously store points in Qdrant by running the synchronous method in an executor.

        This method allows the synchronous upsert operation to be performed without blocking the event loop.

        Args:
            collection_name (str): Name of the Qdrant collection.
            points (List[models.PointStruct]): List of points to store.
        """
        # Get the current event loop
        loop = asyncio.get_running_loop()
        # Run the synchronous storage method in a separate thread to avoid blocking
        await loop.run_in_executor(None, self._store_points_in_qdrant_sync, collection_name, points)

    async def generate_summary_report(self, summary_output_path: str) -> None:
        """
        Generate a summary report from all stored reports and save it asynchronously.

        This method aggregates all individual reports, synthesizes a comprehensive summary
        using an LLM, and saves the summary to the specified path.

        Args:
            summary_output_path (str): Path to save the summary report.
        """
        # Retrieve all reports from the Qdrant reports collection
        all_reports = await self._retrieve_all_reports_async(
            self.reports_collection_name
        )
        if not all_reports:
            # Handle the case where no reports are available
            self.loggers["main"].warning(
                "No reports found in the reports collection to generate a summary."
            )
            summary = "No reports available to generate a summary."
        else:
            self.loggers["llm"].debug(
                "Generating summary from all stored reports"
            )
            # Generate a summary using the ReportPostProcessor, which utilizes an LLM
            summary = await self.report_post_processor.generate_summary(all_reports)
        # Save the generated summary to the specified output path
        await self._save_report_async(summary, summary_output_path)

    async def get_entities_info(
        self, entities: List[Dict[str, str]], context: str
    ) -> Dict[str, Any]:
        """
        Retrieve comprehensive information about a list of entities.

        This method gathers detailed information about each entity, including Wikipedia summaries,
        descriptions, and other relevant metadata, enhancing the depth of entity analysis.

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
                # Create a task to fetch information for each entity
                tasks.append(self.get_entity_info_single(entity, context))
            else:
                # Log an error if the entity dictionary is missing the 'text' field
                self.loggers['wikipedia'].error(f"Entity dictionary missing 'text' field: {entity_dict}")
        # Execute all entity information retrieval tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for entity, result in zip([e.get('text') for e in entities if 'text' in e], results):
            if isinstance(result, dict):
                # Perform entity-level sentiment analysis and add it to the entity info
                entity_sentiment = await self.perform_entity_sentiment_analysis(entity, context)
                result['sentiment'] = entity_sentiment
                wiki_info[entity] = result
            elif isinstance(result, Exception):
                # Log any exceptions that occurred during entity information retrieval
                self.loggers['wikipedia'].error(f"Error fetching info for entity '{entity}': {result}")
            else:
                # Handle cases where no information was found for the entity
                self.loggers['wikipedia'].warning(f"No information found for entity '{entity}'.")
        return wiki_info

    @retry
    async def get_entity_info_single(
        self, entity: str, context: str
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve comprehensive information about a single entity.

        This method performs spell correction, fetches Wikidata information, validates the entity,
        and retrieves detailed Wikipedia summaries and relevant sections.

        Args:
            entity (str): The entity name to retrieve information for.
            context (str): Contextual information for processing.

        Returns:
            Optional[Dict[str, Any]]: Dictionary containing information about the entity or None if not found.
        """
        try:
            # Step 1: Contextual Spell Correction using LLM
            corrected_entity = await self.correct_entity_spelling(entity, context)
            if corrected_entity and corrected_entity != entity:
                self.loggers["wikipedia"].info(
                    f"Corrected entity name from '{entity}' to '{corrected_entity}'"
                )
                entity = corrected_entity

            # Check if the entity information is already cached to avoid redundant API calls
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

            # Validate the entity using its Wikidata ID to ensure relevance and completeness
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
                            # Further validate alignment with context based on keywords
                            context_keywords = set(context.lower().split())
                            description_keywords = set(english_description.lower().split())
                            common_keywords = context_keywords.intersection(description_keywords)
                            if common_keywords:
                                self.loggers["wikipedia"].debug(
                                    f"Entity '{wikidata_id}' aligns with context based on keywords: {common_keywords}"
                                )
                                # Continue processing as the entity is relevant
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

                        # Retrieve site links to get the corresponding Wikipedia title
                        sitelinks = entity_data.get("sitelinks", {})
                        enwiki = sitelinks.get("enwiki", {})
                        wikipedia_title = enwiki.get("title")
                        if not wikipedia_title:
                            self.loggers["wikipedia"].warning(
                                f"No English Wikipedia page found for '{entity}'."
                            )
                            return None

                        # Fetch Wikipedia summary using the summary API
                        summary_api_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{wikipedia_title}"
                        async with self.limiter:
                            async with self.session.get(summary_api_url) as summary_resp:
                                if summary_resp.status == 200:
                                    summary_data = await summary_resp.json()
                                    if summary_data.get("type") == "disambiguation":
                                        # Handle disambiguation pages by suggesting alternative entities
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
                                    aliases = list(set(aliases))  # Remove duplicates

                                    categories = []
                                    if "P910" in claims:  # Additional name property (P910)
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
                        # Handle case where the Wikidata entity is not found
                        self.loggers["wikipedia"].warning(
                            f"Wikidata entity not found for '{entity}'."
                        )
                        return None
                    else:
                        # Log an error if the Wikidata entity request fails with a different status code
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

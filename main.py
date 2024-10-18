import os
import logging
import uuid
import asyncio
from openai import AsyncOpenAI
import spacy
from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime
from functools import wraps
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse
from dotenv import load_dotenv
from dataclasses import dataclass
from enum import Enum
import yaml
import aiohttp
from bs4 import BeautifulSoup
import json
from spellchecker import SpellChecker
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import warnings
from bs4 import GuessedAtParserWarning
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.manifold import TSNE
from collections import defaultdict
import tiktoken  # For accurate token counting

# ------------------------------
# Monkey-Patch BeautifulSoup in wikipedia
# ------------------------------
warnings.filterwarnings("ignore", category=GuessedAtParserWarning)

original_bs_constructor = BeautifulSoup

def patched_bs_constructor(html, *args, **kwargs):
    return original_bs_constructor(html, features="html.parser", *args, **kwargs)

BeautifulSoup = patched_bs_constructor

# ------------------------------
# Logging Configuration
# ------------------------------
logging.basicConfig(
    level=logging.INFO,  # Set to INFO to reduce verbosity; change to DEBUG for more details
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('story_processor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ------------------------------
# Exception Classes
# ------------------------------
class ConfigError(Exception):
    """Custom exception for configuration errors."""
    pass

class ProcessingError(Exception):
    """Custom exception for processing errors."""
    pass

# ------------------------------
# Enum for Model Types
# ------------------------------
class ModelType(Enum):
    EMBEDDING = "embedding"
    CHAT = "chat"

# ------------------------------
# Dataclass for Processing Results
# ------------------------------
@dataclass
class ProcessingResult:
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

# ------------------------------
# Retry Decorator with Exponential Backoff
# ------------------------------
def retry_on_exception(retries: int = 3, base_delay: float = 1.0, factor: float = 2.0):
    """
    Decorator to retry a function upon exception with exponential backoff.
    Supports both synchronous and asynchronous functions.
    """
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            last_exception = None
            delay = base_delay
            for attempt in range(1, retries + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    logger.warning(f"Attempt {attempt} failed: {str(e)}. Retrying in {delay} seconds...")
                    if attempt < retries:
                        await asyncio.sleep(delay)
                        delay *= factor
            logger.error(f"All {retries} attempts failed: {str(last_exception)}")
            raise last_exception

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            last_exception = None
            delay = base_delay
            for attempt in range(1, retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    logger.warning(f"Attempt {attempt} failed: {str(e)}. Retrying in {delay} seconds...")
                    if attempt < retries:
                        time.sleep(delay)
                        delay *= factor
            logger.error(f"All {retries} attempts failed: {str(last_exception)}")
            raise last_exception

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    return decorator

# ------------------------------
# ConfigManager Class
# ------------------------------
@dataclass
class ConfigManager:
    config_dir: str = "config"
    config: Dict[str, Any] = None

    def __post_init__(self):
        self.config_dir = Path(self.config_dir)
        self.config = {}
        self._load_configs()

    def _read_yaml_file(self, filepath: Path) -> Dict[str, Any]:
        try:
            with open(filepath, 'r', encoding='utf-8') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            raise ConfigError(f"Configuration file not found: {filepath}")
        except yaml.YAMLError as e:
            raise ConfigError(f"Error parsing YAML file {filepath}: {str(e)}")
        except Exception as e:
            raise ConfigError(f"Error reading configuration file {filepath}: {str(e)}")

    def _load_configs(self) -> None:
        try:
            config_path = self.config_dir / 'config.yaml'
            if not config_path.exists():
                raise ConfigError(f"Configuration file does not exist: {config_path}")
            self.config = self._read_yaml_file(config_path)
            logging.info("Configuration loaded successfully")
            logging.debug(f"Loaded Configurations: {self.config}")
        except Exception as e:
            logging.error(f"Failed to load configuration: {str(e)}")
            raise

    def get_openai_api_key(self) -> str:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            api_key = self.config.get('openai', {}).get('settings', {}).get('api_key')
        if not api_key:
            raise ConfigError("OpenAI API key not found in environment variables or config.yaml")
        return api_key

    def get_model_config(self, model_type: ModelType) -> Dict[str, Any]:
        try:
            models_config = self.config['openai']['settings']
            if model_type == ModelType.EMBEDDING:
                return {
                    'model': models_config.get('embedding_model', 'text-embedding-ada-002'),
                    'timeout': float(models_config.get('timeout', 30.0))
                }
            elif model_type == ModelType.CHAT:
                return {
                    'model': models_config.get('chat_model', 'gpt-4'),  # Corrected model name
                    'max_tokens': int(models_config.get('max_tokens', 1000)),  # Adjusted max_tokens
                    'temperature': float(models_config.get('temperature', 0.7))
                }
            raise ValueError(f"Unknown model type: {model_type}")
        except KeyError as e:
            raise ConfigError(f"Missing configuration for {model_type}: {str(e)}")

    def get_prompt(self, prompt_type: str, role: str) -> str:
        return self.config.get('prompts', {}).get(prompt_type, {}).get(role, '')

# ------------------------------
# EnhancedEntityProcessor Class
# ------------------------------
class EnhancedEntityProcessor:
    def __init__(self, openai_client: AsyncOpenAI, config_manager: ConfigManager):
        self.wikipedia_cache = {}
        self.wikidata_cache = {}
        self.openai_client = openai_client
        self.config_manager = config_manager
        timeout = aiohttp.ClientTimeout(total=30)  # 30 seconds timeout
        self.session = aiohttp.ClientSession(timeout=timeout)

    async def close(self):
        await self.session.close()

    async def suggest_alternative_entity_name(self, entity: str) -> Optional[str]:
        """
        Use OpenAI to suggest an alternative entity name when Wikipedia page is not found.
        """
        prompt_system = self.config_manager.get_prompt('entity_suggestion', 'system') or "You are an assistant that suggests alternative names for entities."
        prompt_user_template = self.config_manager.get_prompt('entity_suggestion', 'user') or "Given the entity '{entity}', suggest alternative names that might be used to find it on Wikipedia."
        prompt_user = prompt_user_template.format(entity=entity)

        messages = [
            {"role": "system", "content": prompt_system},
            {"role": "user", "content": prompt_user}
        ]

        logging.debug(f"Messages sent to OpenAI for entity suggestion: {messages}")

        try:
            response = await self.openai_client.chat.completions.create(
                model=self.config_manager.get_model_config(ModelType.CHAT)['model'],
                messages=messages,
                max_tokens=50,  # Limit to 50 tokens
                temperature=0.3  # Lower temperature for more deterministic output
            )
            suggestion = response.choices[0].message.content.strip()
            # Extract the first line or suggestion
            suggestion = suggestion.split('\n')[0].replace('.', '').strip()
            # Validate that the suggestion is a plausible entity name
            if suggestion.lower() in ["hello! how can i assist you today?"]:
                logging.error(f"Invalid suggestion received from OpenAI: '{suggestion}'")
                return None
            logging.debug(f"Suggested alternative for '{entity}': '{suggestion}'")
            return suggestion
        except Exception as e:
            logging.error(f"Failed to suggest alternative entity name for '{entity}': {str(e)}")
            return None

    async def get_entity_info(self, entity: str, context: str, retry: bool = True) -> Dict[str, Any]:
        if entity in self.wikipedia_cache:
            logging.debug(f"Entity '{entity}' found in Wikipedia cache")
            return self.wikipedia_cache[entity]

        logging.debug(f"Fetching Wikipedia info for entity '{entity}'")
        try:
            async with self.session.get(f"https://en.wikipedia.org/api/rest_v1/page/summary/{entity}") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    info = {
                        "title": data.get('title', 'No title available'),
                        "summary": data.get('extract', 'No summary available'),
                        "url": data.get('content_urls', {}).get('desktop', {}).get('page', '#'),
                        "categories": data.get('categories', [])[:5] if data.get('categories') else []
                    }
                    self.wikipedia_cache[entity] = info
                    return info
                elif resp.status == 404 and retry:
                    logging.warning(f"Wikipedia page not found for '{entity}'. Attempting to suggest alternative name.")
                    alternative_entity = await self.suggest_alternative_entity_name(entity)
                    if alternative_entity and alternative_entity != entity:
                        return await self.get_entity_info(alternative_entity, context, retry=False)
                    else:
                        logging.warning(f"No Wikipedia page found for '{entity}' and no alternative could be suggested.")
                        return {"error": f"No Wikipedia page found for '{entity}' and no alternative could be suggested."}
                else:
                    logging.error(f"Wikipedia request failed for '{entity}' with status code {resp.status}")
                    return {"error": f"Wikipedia request failed for '{entity}' with status code {resp.status}"}
        except Exception as e:
            logging.error(f"Exception during Wikipedia request for '{entity}': {str(e)}")
            return {"error": f"Exception during Wikipedia request for '{entity}': {str(e)}"}

    async def get_entities_info(self, entities: List[str], context: str) -> Dict[str, Dict[str, Any]]:
        wiki_info = {}
        tasks = []
        for entity in entities:
            tasks.append(self.get_entity_info(entity, context))

        logging.debug(f"Starting asynchronous fetching of entity info for entities: {entities}")
        results = await asyncio.gather(*tasks, return_exceptions=True)
        logging.debug("Completed asynchronous fetching of entity info")

        for entity, result in zip(entities, results):
            if isinstance(result, dict):
                if "error" not in result:
                    wiki_info[entity] = result
                else:
                    logging.warning(f"Failed to retrieve Wikipedia info for '{entity}': {result['error']}")
            elif isinstance(result, Exception):
                logging.error(f"Exception during information lookup for '{entity}': {str(result)}")
            else:
                logging.error(f"Unexpected result type for '{entity}': {type(result)}")
        return wiki_info

# ------------------------------
# Worker Functions
# ------------------------------
_worker_nlp = None
_worker_spell_checker = None

def initialize_worker():
    """
    Initialize resources for worker processes.
    This function is called once per worker process.
    """
    global _worker_nlp
    global _worker_spell_checker
    _worker_nlp = spacy.load("en_core_web_sm")  # Use a lightweight model for efficiency
    _worker_spell_checker = SpellChecker()

def preprocess_text_worker(text: str) -> str:
    """
    Worker function to preprocess text:
    - Spell checking
    - Tokenization
    """
    global _worker_nlp
    global _worker_spell_checker
    if _worker_nlp is None or _worker_spell_checker is None:
        initialize_worker()
    tokens = _worker_nlp(text)
    corrected_tokens = []
    for idx, token in enumerate(tokens):
        if token.is_alpha and not token.ent_type_:
            corrected_word = _worker_spell_checker.correction(token.text)
            if corrected_word is None:
                logging.debug(f"SpellChecker returned None for token '{token.text}' in text: {text[:30]}...")
                corrected_word = token.text
            corrected_tokens.append(corrected_word)
        else:
            token_text = token.text if token.text is not None else ""
            if token.text is None:
                logging.debug(f"Token.text is None at index {idx} in text: {text[:30]}...")
            corrected_tokens.append(token_text)
    corrected_text = ' '.join(corrected_tokens)
    return corrected_text

def extract_and_resolve_entities_worker(text: str) -> List[str]:
    """
    Worker function to extract and resolve entities.
    """
    global _worker_nlp
    if _worker_nlp is None:
        initialize_worker()
    relevant_entity_types = {'PERSON', 'ORG', 'GPE', 'EVENT', 'PRODUCT'}
    doc = _worker_nlp(text)
    entities = list(set([ent.text for ent in doc.ents if ent.label_ in relevant_entity_types]))
    # Resolve entities by choosing the longest name in each group
    entity_groups = defaultdict(list)
    for entity in entities:
        key = ''.join(e.lower() for e in entity if e.isalnum())
        entity_groups[key].append(entity)
    resolved_entities = [max(group, key=len) for group in entity_groups.values()]
    # Ensure all entities are strings
    resolved_entities = [entity if entity is not None else "" for entity in resolved_entities]
    return resolved_entities

# ------------------------------
# ReportPostProcessor Class
# ------------------------------
class ReportPostProcessor:
    def __init__(self, openai_client: AsyncOpenAI, config_manager: ConfigManager):
        self.openai_client = openai_client
        self.config_manager = config_manager
        # Initialize tokenizer
        self.tokenizer = tiktoken.get_encoding("cl100k_base")  # Use the appropriate encoding for your model

    async def refine_report(self, original_story: str, generated_report: str, wikipedia_info: str) -> str:
        prompt_system = self.config_manager.get_prompt('report_refinement', 'system') or "You are a helpful assistant."
        prompt_user_template = self.config_manager.get_prompt('report_refinement', 'user') or "Refine the following report."

        # Prepare the prompt
        prompt = prompt_user_template.format(
            original_story=original_story,
            generated_report=generated_report,
            wikipedia_info=wikipedia_info
        )

        messages = [
            {"role": "system", "content": prompt_system},
            {"role": "user", "content": prompt}
        ]

        logging.debug(f"Messages sent to OpenAI for report refinement: {messages}")

        # Calculate tokens
        total_tokens = sum([len(self.tokenizer.encode(msg["content"])) for msg in messages])
        max_context_length = 8192  # For gpt-4
        buffer_tokens = 1000  # Reserve for the completion and buffer
        available_tokens = max_context_length - total_tokens - buffer_tokens

        if available_tokens <= 0:
            logging.error("Refine report: Not enough tokens available for completion.")
            return generated_report

        max_completion_tokens = min(available_tokens, self.config_manager.get_model_config(ModelType.CHAT)['max_tokens'])

        try:
            response = await self.openai_client.chat.completions.create(
                model=self.config_manager.get_model_config(ModelType.CHAT)['model'],
                messages=messages,
                max_tokens=max_completion_tokens,
                temperature=self.config_manager.get_model_config(ModelType.CHAT)['temperature']
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logging.error(f"Failed to refine report: {str(e)}")
            return generated_report  # Return original report if refinement fails

    def count_tokens(self, text: str) -> int:
        # Using tiktoken for accurate token counting
        return len(self.tokenizer.encode(text))

    async def process_full_report(self, stories: Dict[str, str], generated_report: str, wiki_info: Dict[str, Dict[str, Any]]) -> str:
        sections = generated_report.split("## Story ID:")
        refined_sections = []

        tasks = []
        for section in sections[1:]:  # Skip the first empty section
            story_id = section.split("\n")[0].strip()
            original_story = stories.get(story_id, "")
            story_wiki_info = wiki_info.get(story_id, {})
            wikipedia_info_str = json.dumps(story_wiki_info, indent=2)  # Convert dict to formatted string
            tasks.append(self.refine_report(original_story, f"## Story ID:{section}", wikipedia_info_str))

        refined_sections = await asyncio.gather(*tasks, return_exceptions=True)

        final_sections = []
        for refined in refined_sections:
            if isinstance(refined, str):
                final_sections.append(refined)
            elif isinstance(refined, Exception):
                logging.error(f"Error refining a section: {str(refined)}")
                final_sections.append("## Refinement Failed\n")
            else:
                logging.error(f"Unexpected refinement result type: {type(refined)}")
                final_sections.append("## Refinement Failed\n")

        return "# Refined Analysis Report\n\n" + "\n\n".join(final_sections)

    async def generate_summary(self, reports: List[str]) -> str:
        prompt_system = self.config_manager.get_prompt('summary_generation', 'system') or "You are a helpful assistant."
        prompt_user = self.config_manager.get_prompt('summary_generation', 'user') or "Summarize the following reports."

        # Prepare the prompt
        formatted_reports = '\n\n'.join([f"Report {i+1}:\n{report}" for i, report in enumerate(reports)])
        prompt = prompt_user.format(reports=formatted_reports)

        messages = [
            {"role": "system", "content": prompt_system},
            {"role": "user", "content": prompt}
        ]

        logging.debug(f"Messages sent to OpenAI for summary generation: {messages}")

        # Calculate tokens
        total_tokens = sum([len(self.tokenizer.encode(msg["content"])) for msg in messages])
        max_context_length = 8192  # For gpt-4
        buffer_tokens = 500  # Reserve for the completion and buffer
        available_tokens = max_context_length - total_tokens - buffer_tokens

        if available_tokens <= 0:
            logging.error("Generate summary: Not enough tokens available for completion.")
            return "Summary generation failed due to insufficient context."

        max_completion_tokens = min(available_tokens, 1000)  # Limit to 1000 tokens for summary

        try:
            response = await self.openai_client.chat.completions.create(
                model=self.config_manager.get_model_config(ModelType.CHAT)['model'],
                messages=messages,
                max_tokens=max_completion_tokens,
                temperature=0.5
            )
            summary = response.choices[0].message.content.strip()
            logging.debug("Generated summary report.")
            return summary
        except Exception as e:
            logging.error(f"Failed to generate summary: {str(e)}")
            return "Summary generation failed due to an error."

# ------------------------------
# StoryProcessor Class
# ------------------------------
class StoryProcessor:

    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.stories_collection_name: str = ''
        self.reports_collection_name: str = ''
        self.vector_size: int = 1536
        self.qdrant_client: Optional[QdrantClient] = None
        self.stories: Dict[str, str] = {}
        self.analysis_results: Dict[str, Dict[str, Any]] = {}
        self.openai_client: Optional[AsyncOpenAI] = None
        self.report_post_processor: Optional[ReportPostProcessor] = None
        self.entity_processor: Optional[EnhancedEntityProcessor] = None
        self.process_pool = ProcessPoolExecutor(max_workers=multiprocessing.cpu_count())

    def initialize(self):
        self._initialize_services()

    def _initialize_services(self) -> None:
        try:
            self._setup_openai()
            self._setup_qdrant()
            self._setup_entity_processor()
            logging.info("All services initialized successfully")
        except Exception as e:
            logging.error(f"Service initialization failed: {str(e)}")
            raise

    def _setup_openai(self) -> None:
        load_dotenv()
        api_key = self.config_manager.get_openai_api_key()
        self.openai_client = AsyncOpenAI(api_key=api_key)
        self.report_post_processor = ReportPostProcessor(self.openai_client, self.config_manager)
        logging.info("OpenAI API key set and ReportPostProcessor initialized.")

    def _setup_qdrant(self) -> None:
        qdrant_config = self.config_manager.config.get('qdrant', {}).get('settings', {})
        try:
            self.qdrant_client = QdrantClient(
                host=qdrant_config.get('host', 'localhost'),
                port=int(qdrant_config.get('port', 6333)),
                timeout=self.config_manager.get_model_config(ModelType.EMBEDDING)['timeout']
            )
            logging.info("Qdrant client initialized successfully.")

            # Initialize Stories Collection
            self.stories_collection_name = qdrant_config.get('stories_collection', {}).get('name', 'stories_collection')
            self.vector_size = int(qdrant_config.get('stories_collection', {}).get('vector_size', 1536))
            distance_metric_stories = qdrant_config.get('stories_collection', {}).get('distance', 'COSINE').upper()
            self._initialize_collection(self.stories_collection_name, self.vector_size, distance_metric_stories)

            # Initialize Reports Collection
            self.reports_collection_name = qdrant_config.get('reports_collection', {}).get('name', 'reports_collection')
            reports_vector_size = int(qdrant_config.get('reports_collection', {}).get('vector_size', 1536))
            distance_metric_reports = qdrant_config.get('reports_collection', {}).get('distance', 'COSINE').upper()
            self._initialize_collection(self.reports_collection_name, reports_vector_size, distance_metric_reports)
        except Exception as e:
            logging.error(f"Failed to initialize Qdrant client: {str(e)}")
            raise ProcessingError(f"Qdrant client initialization failed: {str(e)}")
        
    def _initialize_collection(self, collection_name: str, vector_size: int, distance_metric: str) -> None:
        try:
            collections = self.qdrant_client.get_collections()
            if collection_name not in [c.name for c in collections.collections]:
                self.qdrant_client.create_collection(
                    collection_name=collection_name,
                    vectors_config=models.VectorParams(
                        size=vector_size,
                        distance=models.Distance[distance_metric]
                    )
                )
                logging.info(f"Created Qdrant collection: {collection_name}")
            else:
                # Check if the existing collection matches the expected configuration
                collection_info = self.qdrant_client.get_collection(collection_name)
                if collection_info.config.params.vectors.size != vector_size or \
                   collection_info.config.params.vectors.distance != models.Distance[distance_metric]:
                    raise ProcessingError(f"Existing collection '{collection_name}' has mismatched configuration")
                logging.info(f"Qdrant collection '{collection_name}' already exists with correct configuration.")
        except Exception as e:
            raise ProcessingError(f"Failed to initialize Qdrant collection '{collection_name}': {str(e)}")
        
    def _setup_entity_processor(self) -> None:
        if not self.openai_client or not self.config_manager:
            raise ProcessingError("OpenAI client or ConfigManager not initialized.")
        self.entity_processor = EnhancedEntityProcessor(self.openai_client, self.config_manager)
        logging.info("EnhancedEntityProcessor initialized.")

    async def _get_embedding_async(self, text: str) -> List[float]:
        try:
            model_config = self.config_manager.get_model_config(ModelType.EMBEDDING)
            response = await self.openai_client.embeddings.create(
                input=[text],
                model=model_config['model']
            )
            embedding = response.data[0].embedding
            logging.debug(f"Generated embedding for text: {text[:30]}... [truncated]. Embedding size: {len(embedding)}")
            return embedding
        except Exception as e:
            raise ProcessingError(f"Failed to generate embedding: {str(e)}")

    def _prepare_qdrant_points(self, collection_name: str, point_id: str, content: str, wiki_info: Dict[str, Any], embedding: List[float], story_id: str) -> List[models.PointStruct]:
        if not isinstance(embedding, list) or len(embedding) != self.vector_size:
            raise ValueError(f"Invalid embedding: expected list of length {self.vector_size}")

        payload = {
            'report_id': point_id,
            'story_id': story_id,
            'content': content,
            'wiki_info': json.dumps(wiki_info),  # Convert dict to JSON string
            'timestamp': datetime.now().isoformat()
        }
        point = models.PointStruct(
            id=point_id,
            vector=embedding,
            payload=payload
        )
        logging.debug(f"Prepared Qdrant point for ID '{point_id}' in collection '{collection_name}'. Vector size: {len(embedding)}")
        return [point]

    def _generate_report_id(self) -> str:
        # Generate a unique report ID as a standalone UUID
        return str(uuid.uuid4())

    def _store_points_in_qdrant_sync(self, collection_name: str, points: List[models.PointStruct]) -> None:
        try:
            operation_info = self.qdrant_client.upsert(
                collection_name=collection_name,
                points=points
            )
            logging.info(f"Stored {len(points)} point(s) in Qdrant collection '{collection_name}'. Operation ID: {operation_info.operation_id}")
            # Removed wait_for_operation as 'wait=True' in upsert ensures completion
            logging.info(f"Upsert operation completed for collection '{collection_name}'")
        except UnexpectedResponse as e:
            raise ProcessingError(f"Failed to store points in Qdrant: {str(e)}")
        except Exception as e:
            raise ProcessingError(f"An unexpected error occurred while storing points in Qdrant: {str(e)}")

    async def _store_points_in_qdrant(self, collection_name: str, points: List[models.PointStruct]) -> None:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._store_points_in_qdrant_sync, collection_name, points)

    async def _generate_analysis_async(self, content: str, entities: List[str], wiki_info: Dict[str, Dict[str, Any]]) -> str:
        prompt_system = self.config_manager.get_prompt('analysis', 'system') or "You are an assistant that analyzes stories based on their content and entities."
        prompt_user = self.config_manager.get_prompt('analysis', 'user') or "Analyze the following content and entities."

        # Prepare the prompt with necessary truncation
        content_truncated = self.truncate_text(content, 2000)
        entities_truncated = ', '.join(entities[:50])  # Limit to first 50 entities
        wiki_info_truncated = json.dumps(wiki_info, indent=2)[:4000]  # Limit to first 4000 characters

        prompt_user_formatted = f"{prompt_user}\n\nContent: {content_truncated}\n\nEntities: {entities_truncated}\n\nEntity Information:\n{wiki_info_truncated}"

        messages = [
            {"role": "system", "content": prompt_system},
            {"role": "user", "content": prompt_user_formatted}
        ]

        logging.debug(f"Messages sent to OpenAI for analysis: {messages}")

        try:
            model_config = self.config_manager.get_model_config(ModelType.CHAT)
            # Initialize tokenizer
            tokenizer = tiktoken.get_encoding("gpt-4")  # Use the appropriate encoding for your model
            # Calculate tokens
            total_tokens = sum([len(tokenizer.encode(msg["content"])) for msg in messages])
            max_context_length = 8192  # For gpt-4
            buffer_tokens = 500  # Reserve for the completion and buffer
            available_tokens = max_context_length - total_tokens - buffer_tokens

            if available_tokens <= 0:
                logging.error("Generate analysis: Not enough tokens available for completion.")
                raise ProcessingError("Prompt is too long to generate a valid completion.")

            max_completion_tokens = min(available_tokens, model_config['max_tokens'])

            response = await self.openai_client.chat.completions.create(
                model=model_config['model'],
                messages=messages,
                max_tokens=max_completion_tokens,
                temperature=model_config['temperature']
            )
            analysis = response.choices[0].message.content.strip()
            logging.debug(f"Generated analysis for content: {content[:30]}... [truncated]")
            return analysis
        except Exception as e:
            logging.error(f"Failed to generate analysis: {str(e)}")
            raise ProcessingError(f"Failed to generate analysis: {str(e)}")

    def truncate_text(self, text: str, max_chars: int) -> str:
        if len(text) <= max_chars:
            return text
        return text[:max_chars] + "..."

    def _generate_report(self, results: List[ProcessingResult]) -> str:
        report_lines = ["# Analysis Report", ""]
        for result in results:
            if result.success and result.data:
                data = result.data
                story_id = data.get('story_id', 'Unknown')
                report_lines.append(f"## Story ID: {story_id}")

                entities = data.get('entities', [])
                if entities:
                    report_lines.append("### Entities:")
                    report_lines.append("| Entity |")
                    report_lines.append("|--------|")
                    for entity in entities:
                        report_lines.append(f"| {entity} |")
                else:
                    report_lines.append("### Entities: None found")

                analysis = data.get('analysis', 'No analysis available')
                report_lines.append("### Analysis:")
                report_lines.append(analysis)

                report_lines.append("")
            else:
                error_message = result.error if result.error else "Unknown error"
                report_lines.append(f"## Failed to process story: {error_message}")
                report_lines.append("")

        report_content = '\n'.join(report_lines)
        logging.info("Generated analysis report.")
        return report_content

    def _save_report(self, report_content: str, output_path: str) -> None:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report_content)
            logging.info(f"Report saved to {output_path}")
        except Exception as e:
            logging.error(f"Failed to save report to '{output_path}': {str(e)}")

    def _retrieve_all_reports_sync(self, collection_name: str) -> List[str]:
        try:
            all_reports = []
            offset = None
            limit = 100  # Adjust based on expected number of reports

            while True:
                # The scroll method returns a tuple: (points, next_offset)
                search_result, next_offset = self.qdrant_client.scroll(
                    collection_name=collection_name,
                    limit=limit,
                    with_payload=True,
                    with_vectors=False,
                    offset=offset
                )

                if not search_result:
                    break

                for point in search_result:
                    content = point.payload.get('content', '')
                    if content:
                        all_reports.append(content)

                if not next_offset:
                    break

                offset = next_offset

            logging.info(f"Retrieved {len(all_reports)} reports from '{collection_name}' collection.")
            return all_reports
        except Exception as e:
            logging.error(f"Failed to retrieve reports from '{collection_name}' collection: {str(e)}")
            return []

    async def _retrieve_all_reports_async(self, collection_name: str) -> List[str]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._retrieve_all_reports_sync, collection_name)

    def generate_entity_frequency_chart(self, entities: List[str], story_id: str) -> Optional[str]:
        """
        Generate a bar chart for entity frequency and save it as an image.
        Returns the path to the saved image.
        """
        entity_counts = defaultdict(int)
        for entity in entities:
            entity_counts[entity] += 1

        if not entity_counts:
            return None

        entities = list(entity_counts.keys())
        counts = list(entity_counts.values())

        plt.figure(figsize=(10, 6))
        plt.barh(entities, counts, color='skyblue')
        plt.xlabel('Frequency')
        plt.title('Entity Frequency')
        plt.tight_layout()

        chart_path = Path(f"output/visualizations/entity_frequency_{story_id}.png")
        chart_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(chart_path)
        plt.close()
        logging.info(f"Entity frequency chart saved to {chart_path}")
        return str(chart_path)

    def _save_intermediary_data(self, data: Any, path: str, data_type: str) -> None:
        """
        Save intermediary data to the specified path.
        Supports JSON and TXT formats based on data_type.
        """
        output_dir = Path(f"output/{path}")
        output_dir.parent.mkdir(parents=True, exist_ok=True)

        try:
            if data_type in ['entities', 'wiki_info', 'embedding', 'analysis', 'additional_info']:
                with open(output_dir, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2)
                logging.debug(f"Saved {data_type} for '{path}'")
            elif data_type == 'preprocessed_text':
                with open(output_dir, 'w', encoding='utf-8') as f:
                    f.write(data)
                logging.debug(f"Saved preprocessed text for '{path}'")
            else:
                logging.warning(f"Unknown data_type '{data_type}' for saving intermediary data.")
        except Exception as e:
            logging.error(f"Failed to save intermediary data to '{path}': {str(e)}")

    # ------------------------------
    # Additional Visualizations
    # ------------------------------
    def generate_word_cloud(self, processed_results: List[ProcessingResult], output_path: str) -> None:
        """
        Generate a word cloud from entity frequencies across all stories.
        """
        entity_freq = defaultdict(int)
        for result in processed_results:
            if result.success and result.data:
                entities = result.data.get('entities', [])
                for entity in entities:
                    entity_freq[entity] += 1

        if not entity_freq:
            logging.warning("No entities found for word cloud generation.")
            return

        wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(entity_freq)

        plt.figure(figsize=(15, 7.5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.tight_layout(pad=0)

        chart_path = Path(f"output/{output_path}")
        chart_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(chart_path)
        plt.close()
        logging.info(f"Word cloud saved to {chart_path}")

    def generate_entity_distribution(self, processed_results: List[ProcessingResult], output_path: str) -> None:
        """
        Generate a pie chart showing the distribution of entity types.
        """
        entity_type_counts = defaultdict(int)
        for result in processed_results:
            if result.success and result.data:
                wiki_info = result.data.get('wiki_info', {})
                for entity, info in wiki_info.items():
                    # Assuming 'description' contains entity type information
                    description = info.get('summary', '').upper()
                    if 'PERSON' in description:
                        entity_type_counts['PERSON'] += 1
                    elif 'ORGANIZATION' in description or 'ORG' in description:
                        entity_type_counts['ORG'] += 1
                    elif 'LOCATION' in description or 'GPE' in description:
                        entity_type_counts['GPE'] += 1
                    elif 'EVENT' in description:
                        entity_type_counts['EVENT'] += 1
                    elif 'PRODUCT' in description:
                        entity_type_counts['PRODUCT'] += 1
                    else:
                        entity_type_counts['OTHER'] += 1

        if not entity_type_counts:
            logging.warning("No entity types found for distribution chart.")
            return

        labels = list(entity_type_counts.keys())
        sizes = list(entity_type_counts.values())

        plt.figure(figsize=(8, 8))
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        plt.title('Entity Type Distribution')

        chart_path = Path(f"output/{output_path}")
        chart_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(chart_path)
        plt.close()
        logging.info(f"Entity type distribution chart saved to {chart_path}")

    def generate_embeddings_tsne(self, processed_results: List[ProcessingResult], output_path: str) -> None:
        """
        Generate a t-SNE plot for story embeddings.
        """
        embeddings = []
        labels = []
        for result in processed_results:
            if result.success and result.data:
                embedding = result.data.get('embedding', [])
                if embedding:
                    embeddings.append(embedding)
                    labels.append(result.data.get('story_id', 'Unknown'))

        if not embeddings:
            logging.warning("No embeddings found for t-SNE visualization.")
            return

        try:
            tsne = TSNE(n_components=2, random_state=42, perplexity=5, n_iter=300)
            embeddings_2d = tsne.fit_transform(embeddings)

            plt.figure(figsize=(12, 8))
            plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], s=50, alpha=0.7)

            for i, label in enumerate(labels):
                plt.annotate(label, (embeddings_2d[i, 0], embeddings_2d[i, 1]))

            plt.title('t-SNE Visualization of Story Embeddings')
            plt.xlabel('Dimension 1')
            plt.ylabel('Dimension 2')
            plt.tight_layout()

            chart_path = Path(f"output/{output_path}")
            chart_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(chart_path)
            plt.close()
            logging.info(f"t-SNE embeddings plot saved to {chart_path}")
        except Exception as e:
            logging.error(f"Failed to generate t-SNE plot: {str(e)}")

    def generate_story_length_histogram(self, processed_results: List[ProcessingResult], output_path: str) -> None:
        """
        Generate a histogram of story lengths.
        """
        story_lengths = []
        for result in processed_results:
            if result.success and result.data:
                analysis = result.data.get('analysis', '')
                word_count = len(analysis.split())
                story_lengths.append(word_count)

        if not story_lengths:
            logging.warning("No story lengths found for histogram.")
            return

        plt.figure(figsize=(10, 6))
        plt.hist(story_lengths, bins=10, color='skyblue', edgecolor='black')
        plt.xlabel('Number of Words')
        plt.ylabel('Number of Stories')
        plt.title('Distribution of Story Lengths')
        plt.tight_layout()

        chart_path = Path(f"output/{output_path}")
        chart_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(chart_path)
        plt.close()
        logging.info(f"Story length distribution histogram saved to {chart_path}")

    async def process_story_async(self, story_id: str, content: str) -> ProcessingResult:
        try:
            logging.debug(f"Starting preprocessing for story '{story_id}'")

            corrected_content = await asyncio.get_event_loop().run_in_executor(
                self.process_pool, preprocess_text_worker, content
            )

            resolved_entities = await asyncio.get_event_loop().run_in_executor(
                self.process_pool, extract_and_resolve_entities_worker, corrected_content
            )

            wiki_info = await self.entity_processor.get_entities_info(resolved_entities, corrected_content)

            logging.debug(f"Generating embedding for story '{story_id}'")
            embedding = await self._get_embedding_async(corrected_content)
            logging.debug(f"Embedding generated for story '{story_id}'")

            # Save Embedding
            self._save_intermediary_data(
                data=embedding,
                path=f"intermediary/embeddings/{story_id}.json",
                data_type='embedding'
            )

            # Generate a UUID for the story's point_id
            story_uuid = self._generate_report_id()
            points = self._prepare_qdrant_points(self.stories_collection_name, story_uuid, corrected_content, wiki_info, embedding, story_id)

            await self._store_points_in_qdrant(self.stories_collection_name, points)

            logging.debug(f"Generating analysis for story '{story_id}'")
            analysis = await self._generate_analysis_async(corrected_content, resolved_entities, wiki_info)
            logging.debug(f"Analysis generated for story '{story_id}'")

            # Save Analysis
            self._save_intermediary_data(
                data=analysis,
                path=f"intermediary/analysis/{story_id}.json",
                data_type='analysis'
            )

            # Generate entity frequency chart
            chart_path = self.generate_entity_frequency_chart(resolved_entities, story_id)

            result = {
                'story_id': story_id,
                'entities': resolved_entities,
                'wiki_info': wiki_info,
                'analysis': analysis,
                'timestamp': datetime.now().isoformat(),
                'entity_frequency_chart': chart_path  # Path to the generated chart
            }

            logging.info(f"Successfully processed story '{story_id}'")
            return ProcessingResult(success=True, data=result)
        except ProcessingError as pe:
            logging.error(f"Failed to process story '{story_id}': {str(pe)}")
            return ProcessingResult(success=False, error=str(pe))
        except Exception as e:
            logging.error(f"Failed to process story '{story_id}': {str(e)}")
            return ProcessingResult(success=False, error=str(e))

    async def process_stories_async(self, stories_dir: str, output_path: str, summary_output_path: str) -> None:
        try:
            self._load_stories(stories_dir)

            # Limit the number of concurrent tasks to prevent overwhelming system resources
            semaphore = asyncio.Semaphore(multiprocessing.cpu_count() * 2)  # Adjust as needed

            async def semaphore_wrapper(story_id, content):
                async with semaphore:
                    return await self.process_story_async(story_id, content)

            tasks = []
            for story_id, content in self.stories.items():
                logging.info(f"Processing story: {story_id}")
                tasks.append(semaphore_wrapper(story_id, content))

            results = await asyncio.gather(*tasks, return_exceptions=True)

            processed_results = []
            wiki_info = {}
            for result in results:
                if isinstance(result, ProcessingResult):
                    processed_results.append(result)
                    if result.success and result.data:
                        wiki_info[result.data['story_id']] = result.data.get('wiki_info', {})
                elif isinstance(result, Exception):
                    logging.error(f"Error processing story: {str(result)}")
                    processed_results.append(ProcessingResult(success=False, error=str(result)))
                else:
                    logging.error(f"Unexpected result type: {type(result)}")
                    processed_results.append(ProcessingResult(success=False, error="Unexpected result type"))

            initial_report = self._generate_report(processed_results)

            # Post-process the report
            refined_report = await self.report_post_processor.process_full_report(self.stories, initial_report, wiki_info)

            # Save the refined report
            self._save_report(refined_report, output_path)

            # Store individual refined reports in the reports_collection
            for result in processed_results:
                if result.success and result.data:
                    story_id = result.data['story_id']
                    report_content = result.data['analysis']
                    wiki_info_content = json.dumps(result.data['wiki_info'], indent=2)
                    report_id = self._generate_report_id()  # Generate standalone UUID
                    embedding = await self._get_embedding_async(report_content)
                    report_points = self._prepare_qdrant_points(
                        self.reports_collection_name,
                        report_id,
                        report_content,
                        result.data['wiki_info'],
                        embedding,
                        story_id  # Pass story_id
                    )
                    await self._store_points_in_qdrant(self.reports_collection_name, report_points)
                    logging.info(f"Refined report for story '{story_id}' stored in '{self.reports_collection_name}' collection.")

            # Retrieve all reports from the reports collection
            logging.debug("Retrieving all reports from the reports collection for summary generation")
            all_reports = await self._retrieve_all_reports_async(self.reports_collection_name)

            if not all_reports:
                logging.warning("No reports found in the reports collection to generate a summary.")
                summary = "No reports available to generate a summary."
            else:
                # Generate summary from all reports
                logging.debug("Generating summary from all stored reports")
                summary = await self.report_post_processor.generate_summary(all_reports)

            # Save the summary report
            self._save_report(summary, summary_output_path)
            logging.info(f"Summary report saved to {summary_output_path}")

            # Generate additional visualizations
            self.generate_word_cloud(processed_results, "visualizations/word_cloud.png")
            self.generate_entity_distribution(processed_results, "visualizations/entity_distribution.png")
            self.generate_embeddings_tsne(processed_results, "visualizations/embeddings_tsne.png")
            self.generate_story_length_histogram(processed_results, "visualizations/story_length_histogram.png")

            logging.info(f"Processing complete. Refined reports saved to '{self.reports_collection_name}' and summary saved to {summary_output_path}")
        except Exception as e:
            logging.error(f"Failed to process stories: {str(e)}", exc_info=True)
        finally:
            # Close the EnhancedEntityProcessor session and Qdrant client
            if self.entity_processor:
                await self.entity_processor.close()
            if self.qdrant_client:
                self.qdrant_client.close()  # Removed 'await'
                logging.info("Qdrant client connection closed.")
            # Shut down the process pool
            if self.process_pool:
                self.process_pool.shutdown(wait=True)
                logging.info("Process pool shut down successfully.")

    def _load_stories(self, stories_dir: str) -> None:
        stories_path = Path(stories_dir)
        if not stories_path.exists() or not stories_path.is_dir():
            raise ProcessingError(f"Stories directory not found: {stories_dir}")

        self.stories = {}
        for story_file in stories_path.glob('*.md'):
            try:
                with open(story_file, 'r', encoding='utf-8') as f:
                    self.stories[story_file.stem] = f.read()
                    logging.debug(f"Loaded story '{story_file.stem}'")
            except Exception as e:
                logging.error(f"Failed to read story file '{story_file}': {str(e)}")
                continue

        if not self.stories:
            raise ProcessingError("No stories found to process.")

        logging.info(f"Loaded {len(self.stories)} story/stories for processing.")

# ------------------------------
# Main Function
# ------------------------------
async def main():
    try:
        config_manager = ConfigManager()
        processor = StoryProcessor(config_manager)
        processor.initialize()
        await processor.process_stories_async(
            stories_dir="stories",
            output_path="output/report.md",
            summary_output_path="output/summary_report.md"
        )
    except Exception as e:
        logger.error(f"Application error: {str(e)}", exc_info=True)
        raise

# ------------------------------
# Entry Point
# ------------------------------
if __name__ == "__main__":
    # Use 'spawn' start method for compatibility, especially on Windows
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        # The start method has already been set; ignore the error
        pass
    asyncio.run(main())

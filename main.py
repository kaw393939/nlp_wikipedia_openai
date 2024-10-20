import os
import logging
import uuid
import asyncio
import time
from openai import AsyncOpenAI
import spacy
from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime, timedelta
from functools import wraps
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Record
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
import tiktoken
import aiofiles
import plotly.express as px
import numpy as np  # Ensure NumPy is imported

import threading

# Suppress specific warnings
warnings.filterwarnings("ignore", category=GuessedAtParserWarning)

# Patch BeautifulSoup to use 'html.parser' by default
original_bs_constructor = BeautifulSoup

def patched_bs_constructor(html, *args, **kwargs):
    return original_bs_constructor(html, features="html.parser", *args, **kwargs)

BeautifulSoup = patched_bs_constructor

# Custom Exceptions
class ConfigError(Exception):
    pass

class ProcessingError(Exception):
    pass

# Enum for Model Types
class ModelType(Enum):
    EMBEDDING = "embedding"
    CHAT = "chat"

# Dataclass for Processing Results
@dataclass
class ProcessingResult:
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

# Retry Decorators
def retry_async(retries=3, base_delay=1.0, factor=2.0):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            attempt = 0
            delay = base_delay
            while attempt < retries:
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    attempt += 1
                    if attempt >= retries:
                        raise
                    logger = args[0].loggers.get('llm', logging.getLogger('llm'))
                    logger.warning(f"Attempt {attempt} failed: {e}. Retrying in {delay} seconds...")
                    await asyncio.sleep(delay)
                    delay *= factor
        return wrapper
    return decorator

def retry_sync(retries=3, base_delay=1.0, factor=2.0):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            attempt = 0
            delay = base_delay
            while attempt < retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    attempt += 1
                    if attempt >= retries:
                        raise
                    logger = args[0].loggers.get('llm', logging.getLogger('llm'))
                    logger.warning(f"Attempt {attempt} failed: {e}. Retrying in {delay} seconds...")
                    time.sleep(delay)
                    delay *= factor
        return wrapper
    return decorator

# Config Manager Class
class ConfigManager:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, config_dir: str = "config"):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(ConfigManager, cls).__new__(cls)
                    cls._instance.config_dir = Path(config_dir)
                    cls._instance.config = {}
                    cls._instance._load_configs()
        return cls._instance

    def _read_yaml_file(self, filepath: Path) -> Dict[str, Any]:
        try:
            with open(filepath, 'r', encoding='utf-8', errors='replace') as file:
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
            required_sections = ['openai', 'qdrant', 'prompts', 'logging', 'retry', 'aiohttp', 'process_pool', 'paths', 'concurrency', 'analysis']
            for section in required_sections:
                if section not in self.config:
                    raise ConfigError(f"Missing required configuration section: '{section}'")
            if 'settings' not in self.config['openai'] or 'api_key' not in self.config['openai']['settings']:
                raise ConfigError("Missing 'api_key' in 'openai.settings' section")
            logging.getLogger('main').info("Configuration loaded and validated successfully")
            logging.getLogger('main').debug(f"Loaded Configurations: {self.config}")
        except Exception as e:
            logging.getLogger('errors').error(f"Failed to load configuration: {str(e)}")
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
                    'timeout': float(models_config.get('timeout', 30.0)),
                    'context_length': models_config.get('context_length', 8192)
                }
            elif model_type == ModelType.CHAT:
                return {
                    'model': models_config.get('chat_model', 'gpt-4'),
                    'max_tokens': int(models_config.get('max_tokens', 4096)),
                    'temperature': float(models_config.get('temperature', 0.7)),
                    'context_length': models_config.get('context_length', 8192)
                }
            raise ValueError(f"Unknown model type: {model_type}")
        except KeyError as e:
            raise ConfigError(f"Missing configuration for {model_type}: {str(e)}")

    def get_prompt(self, prompt_type: str, role: str) -> str:
        return self.config.get('prompts', {}).get(prompt_type, {}).get(role, '')

    def get_logging_config(self) -> Dict[str, Any]:
        return self.config.get('logging', {})

    def get_retry_config(self) -> Dict[str, Any]:
        return self.config.get('retry', {})

    def get_aiohttp_config(self) -> Dict[str, Any]:
        return self.config.get('aiohttp', {})

    def get_process_pool_config(self) -> Dict[str, Any]:
        return self.config.get('process_pool', {})

    def get_paths_config(self) -> Dict[str, Any]:
        return self.config.get('paths', {})

    def get_concurrency_config(self) -> Dict[str, Any]:
        return self.config.get('concurrency', {})

    def get_analysis_config(self) -> Dict[str, Any]:
        return self.config.get('analysis', {})

# Logger Factory Class
class LoggerFactory:
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.loggers = {}
        self._create_loggers()

    def _create_loggers(self):
        # Load logging configuration from logging.yaml
        logging_config = self.config_manager.get_logging_config()
        level_str = logging_config.get('level', 'INFO').upper()
        level = getattr(logging, level_str, logging.INFO)
        log_format = logging_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        formatter = logging.Formatter(log_format)

        # Define logger configurations
        logger_definitions = {
            'main': {
                'level': level,
                'handlers': [
                    ('main_file', logging.FileHandler(logging_config.get('handlers', {}).get('main_file', 'logs/main.log'))),
                    ('stream', logging.StreamHandler())
                ],
            },
            'wikipedia': {
                'level': logging_config.get('handlers', {}).get('wikipedia_level', 'DEBUG').upper(),
                'handlers': [
                    ('wikipedia_file', logging.FileHandler(logging_config.get('handlers', {}).get('wikipedia_file', 'logs/wikipedia.log')))
                ],
            },
            'llm': {
                'level': logging_config.get('handlers', {}).get('llm_level', 'DEBUG').upper(),
                'handlers': [
                    ('llm_file', logging.FileHandler(logging_config.get('handlers', {}).get('llm_file', 'logs/llm.log')))
                ],
            },
            'errors': {
                'level': logging.ERROR,
                'handlers': [
                    ('error_file', logging.FileHandler(logging_config.get('handlers', {}).get('error_file', 'logs/errors.log')))
                ],
            }
        }

        # Create and configure each logger
        for logger_name, props in logger_definitions.items():
            logger = logging.getLogger(logger_name)
            logger.setLevel(props['level'])
            for handler_name, handler in props['handlers']:
                if isinstance(handler, logging.FileHandler):
                    log_file_path = Path(handler.baseFilename)
                    try:
                        log_file_path.parent.mkdir(parents=True, exist_ok=True)
                    except Exception:
                        continue
                handler.setLevel(props['level'])
                handler.setFormatter(formatter)
                logger.addHandler(handler)
            logger.propagate = False
            self.loggers[logger_name] = logger

    def get_logger(self, name: str) -> logging.Logger:
        return self.loggers.get(name, logging.getLogger('main'))

# Enhanced Entity Processor Class
class EnhancedEntityProcessor:
    def __init__(self, openai_client: AsyncOpenAI, config_manager: ConfigManager, loggers: Dict[str, logging.Logger]):
        self.wikipedia_cache = {}
        self.wikidata_cache = {}
        self.openai_client = openai_client
        self.config_manager = config_manager
        self.loggers = loggers
        aiohttp_config = self.config_manager.get_aiohttp_config()
        timeout_seconds = aiohttp_config.get('timeout', 30)
        max_connections = aiohttp_config.get('max_connections', 10)  # Limit to 10 concurrent connections
        connector = aiohttp.TCPConnector(limit=max_connections)
        timeout = aiohttp.ClientTimeout(total=timeout_seconds)
        self.session = aiohttp.ClientSession(connector=connector, timeout=timeout)
        self.rate_limit = aiohttp_config.get('rate_limit', 5)  # Max 5 requests per second
        self.semaphore = asyncio.Semaphore(self.rate_limit)
        self.token_bucket = asyncio.Queue(maxsize=self.rate_limit)
        asyncio.create_task(self._fill_token_bucket())

    async def _fill_token_bucket(self):
        while True:
            if not self.token_bucket.full():
                await self.token_bucket.put(1)
            await asyncio.sleep(1 / self.rate_limit)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.close()

    async def close(self):
        await self.session.close()

    @retry_async()
    async def suggest_alternative_entity_name(self, entity: str) -> Optional[str]:
        prompt_system = self.config_manager.get_prompt('entity_suggestion', 'system') or "You are an assistant that suggests alternative names for entities."
        prompt_user_template = self.config_manager.get_prompt('entity_suggestion', 'user') or "Given the entity '{entity}', suggest alternative names that might be used to find it on Wikipedia."
        prompt_user = prompt_user_template.format(entity=entity)
        messages = [
            {"role": "system", "content": prompt_system},
            {"role": "user", "content": prompt_user}
        ]
        self.loggers['llm'].debug(f"Messages sent to OpenAI for entity suggestion: {messages}")
        try:
            response = await self.openai_client.chat.completions.create(
                model=self.config_manager.get_model_config(ModelType.CHAT)['model'],
                messages=messages,
                max_tokens=50,
                temperature=0.3
            )
            if not response.choices or not response.choices[0].message:
                self.loggers['llm'].error("OpenAI response is missing choices or messages.")
                return None
            suggestion = response.choices[0].message.content.strip()
            suggestion = suggestion.split('\n')[0].replace('.', '').strip()
            if suggestion.lower() in ["hello! how can i assist you today?"]:
                self.loggers['llm'].error(f"Invalid suggestion received from OpenAI: '{suggestion}'")
                return None
            self.loggers['wikipedia'].debug(f"Suggested alternative for '{entity}': '{suggestion}'")
            return suggestion
        except Exception as e:
            self.loggers['llm'].error(f"Failed to suggest alternative entity name for '{entity}': {str(e)}")
            return None

    @retry_async()
    async def get_wikidata_id(self, entity: str) -> Optional[str]:
        """
        Fetch the Wikidata ID for a given entity using the Wikidata API.
        """
        await self.token_bucket.get()
        wikidata_api_url = "https://www.wikidata.org/w/api.php"
        params = {
            "action": "wbsearchentities",
            "format": "json",
            "language": "en",
            "search": entity
        }
        try:
            async with self.session.get(wikidata_api_url, params=params) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if data.get('search'):
                        return data['search'][0].get('id')
            self.loggers['wikipedia'].warning(f"No Wikidata ID found for entity '{entity}'.")
            return None
        except Exception as e:
            self.loggers['wikipedia'].error(f"Error fetching Wikidata ID for '{entity}': {e}")
            return None

    @retry_async()
    async def validate_entity(self, wikidata_id: str) -> bool:
        """
        Validate the entity by checking essential properties on Wikidata.
        """
        await self.token_bucket.get()
        wikidata_api_url = f"https://www.wikidata.org/wiki/Special:EntityData/{wikidata_id}.json"
        try:
            async with self.session.get(wikidata_api_url) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    entity_data = data.get('entities', {}).get(wikidata_id, {})
                    claims = entity_data.get('claims', {})
                    # Example: Check if the entity has an occupation (P106)
                    if 'P106' in claims:
                        return True
            self.loggers['wikipedia'].warning(f"Entity '{wikidata_id}' lacks essential properties.")
            return False
        except Exception as e:
            self.loggers['wikipedia'].error(f"Error validating entity '{wikidata_id}': {e}")
            return False

    @retry_async()
    async def get_entity_sections(self, title: str) -> Dict[str, Any]:
        """
        Fetch specific sections from a Wikipedia page.
        """
        await self.token_bucket.get()
        sections_api_url = f"https://en.wikipedia.org/api/rest_v1/page/mobile-sections/{title}"
        try:
            async with self.session.get(sections_api_url) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    sections = data.get('sections', [])
                    section_contents = {}
                    for section in sections:
                        section_title = section.get('title', 'No Title')
                        section_text = section.get('text', '')
                        if section_text:
                            section_contents[section_title] = section_text
                    return section_contents
                else:
                    self.loggers['wikipedia'].error(f"Wikipedia sections request failed for '{title}' with status code {resp.status}")
                    return {}
        except Exception as e:
            self.loggers['wikipedia'].error(f"Exception during Wikipedia sections request for '{title}': {e}")
            return {}

    @retry_async()
    async def get_entity_info(self, entity: str, context: str, retry: bool = True) -> Dict[str, Any]:
        if entity in self.wikipedia_cache:
            self.loggers['wikipedia'].debug(f"Entity '{entity}' found in Wikipedia cache")
            return self.wikipedia_cache[entity]
        self.loggers['wikipedia'].debug(f"Fetching Wikipedia info for entity '{entity}'")
        try:
            wikidata_id = await self.get_wikidata_id(entity)
            if not wikidata_id:
                return {"error": f"No Wikidata ID found for '{entity}'."}
            is_valid = await self.validate_entity(wikidata_id)
            if not is_valid:
                self.loggers['wikipedia'].warning(f"Entity '{entity}' with Wikidata ID '{wikidata_id}' is invalid.")
                return {"error": f"Entity '{entity}' is invalid based on Wikidata properties."}

            # Fetch Wikidata entity data
            wikidata_api_url = f"https://www.wikidata.org/wiki/Special:EntityData/{wikidata_id}.json"
            async with self.session.get(wikidata_api_url) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    entities = data.get('entities', {})
                    entity_data = entities.get(wikidata_id, {})
                    sitelinks = entity_data.get('sitelinks', {})
                    enwiki = sitelinks.get('enwiki', {})
                    wikipedia_title = enwiki.get('title')
                    if not wikipedia_title:
                        self.loggers['wikipedia'].warning(f"No English Wikipedia page found for '{entity}'.")
                        return {"error": f"No English Wikipedia page found for '{entity}'."}

                    # Fetch Wikipedia summary
                    await self.token_bucket.get()
                    async with self.session.get(f"https://en.wikipedia.org/api/rest_v1/page/summary/{wikipedia_title}") as summary_resp:
                        if summary_resp.status == 200:
                            summary_data = await summary_resp.json()
                            # Check if page is a disambiguation page
                            if 'disambiguation' in [cat.get('title', '').lower() for cat in summary_data.get('categories', [])]:
                                self.loggers['wikipedia'].warning(f"Entity '{entity}' leads to a disambiguation page.")
                                alternative_entity = await self.suggest_alternative_entity_name(entity)
                                if alternative_entity and alternative_entity != entity:
                                    return await self.get_entity_info(alternative_entity, context, retry=False)
                                else:
                                    self.loggers['wikipedia'].warning(f"No suitable alternative could be suggested for disambiguated entity '{entity}'.")
                                    return {"error": f"Entity '{entity}' is ambiguous and no suitable alternative was found."}

                            info = {
                                "wikidata_id": wikidata_id,
                                "title": summary_data.get('title', 'No title available'),
                                "summary": summary_data.get('extract', 'No summary available'),
                                "url": summary_data.get('content_urls', {}).get('desktop', {}).get('page', '#'),
                                "categories": [cat.get('title', '') for cat in summary_data.get('categories', [])][:5],
                                "type": summary_data.get('type', 'UNKNOWN'),
                                "aliases": [alias.get('value') for alias in entity_data.get('aliases', {}).get('en', [])]
                            }

                            # Fetch Wikipedia sections
                            sections = await self.get_entity_sections(wikipedia_title)
                            info.update({"sections": sections})

                            # Enhanced Entity Resolution: Cross-reference aliases to prevent misspellings
                            if not info['aliases']:
                                info['aliases'] = [entity]
                            else:
                                info['aliases'] = list(set(info['aliases'] + [entity]))

                            # Cache the complete info
                            self.wikipedia_cache[entity] = info
                            self.loggers['wikipedia'].debug(f"Retrieved and cached Wikipedia info for '{entity}': {info}")
                            return info
                        elif summary_resp.status == 404 and retry:
                            self.loggers['wikipedia'].warning(f"Wikipedia page not found for '{wikipedia_title}'. Attempting to suggest alternative name.")
                            alternative_entity = await self.suggest_alternative_entity_name(entity)
                            if alternative_entity and alternative_entity != entity:
                                return await self.get_entity_info(alternative_entity, context, retry=False)
                            else:
                                self.loggers['wikipedia'].warning(f"No Wikipedia page found for '{entity}' and no alternative could be suggested.")
                                return {"error": f"No Wikipedia page found for '{entity}' and no alternative could be suggested."}
                        else:
                            self.loggers['wikipedia'].error(f"Wikipedia summary request failed for '{wikipedia_title}' with status code {summary_resp.status}")
                            return {"error": f"Wikipedia summary request failed for '{wikipedia_title}' with status code {summary_resp.status}"}
        except Exception as e:
            self.loggers['wikipedia'].error(f"Exception during Wikipedia request for '{entity}': {e}")
            return {"error": f"Exception during Wikipedia request for '{entity}': {e}"}

    @retry_async()
    async def get_entities_info(self, entities: List[Dict[str, str]], context: str) -> Dict[str, Dict[str, Any]]:
        wiki_info = {}
        tasks = []
        for entity in entities:
            tasks.append(self.get_entity_info(entity['text'], context))
        self.loggers['wikipedia'].debug(f"Starting asynchronous fetching of entity info for entities: {entities}")
        results = await asyncio.gather(*tasks, return_exceptions=True)
        self.loggers['wikipedia'].debug("Completed asynchronous fetching of entity info")
        for entity, result in zip(entities, results):
            if isinstance(result, dict):
                if "error" not in result:
                    wiki_info[entity['text']] = result
                else:
                    self.loggers['wikipedia'].warning(f"Failed to retrieve Wikipedia info for '{entity['text']}': {result['error']}")
            elif isinstance(result, Exception):
                self.loggers['wikipedia'].error(f"Exception during information lookup for '{entity['text']}': {str(result)}")
            else:
                self.loggers['wikipedia'].error(f"Unexpected result type for '{entity['text']}': {type(result)}")
        return wiki_info

# Report Post Processor Class
class ReportPostProcessor:
    def __init__(self, openai_client: AsyncOpenAI, config_manager: ConfigManager, loggers: Dict[str, logging.Logger]):
        self.openai_client = openai_client
        self.config_manager = config_manager
        self.loggers = loggers
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    @retry_async()
    async def refine_report(self, original_story: str, generated_report: str, wikipedia_info: str, wikipedia_sections: str) -> str:
        prompt_system = self.config_manager.get_prompt('report_refinement', 'system') or "You are a helpful assistant."
        prompt_user_template = self.config_manager.get_prompt('report_refinement', 'user') or "Refine the following report."
        prompt = prompt_user_template.format(
            original_story=original_story,
            generated_report=generated_report,
            wikipedia_info=wikipedia_info,
            wikipedia_sections=wikipedia_sections
        )
        messages = [
            {"role": "system", "content": prompt_system},
            {"role": "user", "content": prompt}
        ]
        self.loggers['llm'].debug(f"Messages sent to OpenAI for report refinement: {messages}")
        input_tokens = sum([len(self.tokenizer.encode(msg["content"])) for msg in messages])
        model_config = self.config_manager.get_model_config(ModelType.CHAT)
        max_context_length = model_config['context_length']
        buffer_tokens = self.config_manager.get_retry_config().get('buffer_tokens', 1000)
        available_tokens = max_context_length - input_tokens - buffer_tokens
        self.loggers['llm'].debug(f"Total tokens in prompt: {input_tokens}")
        self.loggers['llm'].debug(f"Available tokens for completion: {available_tokens}")
        if available_tokens <= 0:
            self.loggers['llm'].error("Refine report: Not enough tokens available for completion.")
            return generated_report
        max_completion_tokens = min(model_config['max_tokens'], available_tokens)
        try:
            response = await self.openai_client.chat.completions.create(
                model=model_config['model'],
                messages=messages,
                max_tokens=max_completion_tokens,
                temperature=model_config['temperature']
            )
            if not response.choices or not response.choices[0].message:
                self.loggers['llm'].error("OpenAI response is missing choices or messages.")
                return generated_report
            refined_report = response.choices[0].message.content.strip()
            self.loggers['llm'].debug(f"Refined report generated.")
            return refined_report
        except Exception as e:
            self.loggers['llm'].error(f"Failed to refine report: {str(e)}")
            return generated_report

    @retry_async()
    async def generate_summary(self, reports: List[str]) -> str:
        prompt_system = self.config_manager.get_prompt('summary_generation', 'system') or "You are an expert AI assistant specializing in synthesizing multiple reports into a cohesive, comprehensive, and insightful summary. Your role is to distill essential points from each report to create a detailed and informative overview."
        prompt_user = self.config_manager.get_prompt('summary_generation', 'user') or "Please generate a comprehensive and in-depth summary based on the following refined reports:"
        max_reports_tokens = 3000
        formatted_reports = []
        total_tokens = 0
        for i, report in enumerate(reports):
            report_tokens = len(self.tokenizer.encode(report))
            if total_tokens + report_tokens > max_reports_tokens:
                break
            formatted_reports.append(f"Report {i+1}:\n{report}")
            total_tokens += report_tokens
        if not formatted_reports:
            return "No reports available to summarize due to token limit constraints."
        prompt = prompt_user.format(reports='\n\n'.join(formatted_reports))
        messages = [
            {"role": "system", "content": prompt_system},
            {"role": "user", "content": prompt}
        ]
        self.loggers['llm'].debug(f"Messages sent to OpenAI for summary generation: {messages}")
        input_tokens = sum([len(self.tokenizer.encode(msg["content"])) for msg in messages])
        model_config = self.config_manager.get_model_config(ModelType.CHAT)
        max_context_length = model_config['context_length']
        buffer_tokens = self.config_manager.get_retry_config().get('summary_buffer_tokens', 500)
        available_tokens = max_context_length - input_tokens - buffer_tokens
        self.loggers['llm'].debug(f"Total tokens in summary prompt: {input_tokens}")
        self.loggers['llm'].debug(f"Available tokens for summary completion: {available_tokens}")
        if available_tokens <= 0:
            self.loggers['llm'].error("Generate summary: Not enough tokens available for completion.")
            return "Summary generation failed due to insufficient context. Too many reports to summarize."
        max_completion_tokens = min(self.config_manager.get_retry_config().get('summary_max_tokens', 1000), available_tokens)
        try:
            response = await self.openai_client.chat.completions.create(
                model=model_config['model'],
                messages=messages,
                max_tokens=max_completion_tokens,
                temperature=model_config['temperature']
            )
            if not response.choices or not response.choices[0].message:
                self.loggers['llm'].error("OpenAI response is missing choices or messages.")
                return "Summary generation failed due to an error."
            summary = response.choices[0].message.content.strip()
            self.loggers['llm'].debug("Generated summary report.")
            if len(formatted_reports) < len(reports):
                summary += f"\n\nNote: This summary is based on {len(formatted_reports)} out of {len(reports)} total reports due to token limitations."
            return summary
        except Exception as e:
            self.loggers['llm'].error(f"Failed to generate summary: {str(e)}")
            return "Summary generation failed due to an error."

    @retry_async()
    async def generate_analysis(self, content: str, entities: List[Dict[str, str]], wiki_info: Dict[str, Dict[str, Any]]) -> str:
        prompt_system = self.config_manager.get_prompt('analysis', 'system') or "You are an assistant that analyzes stories based on their content and entities."
        prompt_user = self.config_manager.get_prompt('analysis', 'user') or "Analyze the following content and entities."
        analysis_config = self.config_manager.get_analysis_config()
        content_truncated = self.truncate_text(content, analysis_config.get('content_max_chars', 5000))
        entities_truncated = ', '.join([entity['text'] for entity in entities[:analysis_config.get('entities_limit', 100)]])
        wiki_info_truncated = json.dumps(wiki_info, indent=2)[:analysis_config.get('wiki_info_max_chars', 8000)]
        prompt_user_formatted = f"{prompt_user}\n\nContent: {content_truncated}\n\nEntities: {entities_truncated}\n\nEntity Information:\n{wiki_info_truncated}"
        messages = [
            {"role": "system", "content": prompt_system},
            {"role": "user", "content": prompt_user_formatted}
        ]
        self.loggers['llm'].debug(f"Messages sent to OpenAI for analysis: {messages}")
        input_tokens = sum([len(self.tokenizer.encode(msg["content"])) for msg in messages])
        model_config = self.config_manager.get_model_config(ModelType.CHAT)
        max_context_length = model_config['context_length']
        buffer_tokens = self.config_manager.get_retry_config().get('buffer_tokens', 500)
        available_tokens = max_context_length - input_tokens - buffer_tokens
        self.loggers['llm'].debug(f"Total tokens in analysis prompt: {input_tokens}")
        self.loggers['llm'].debug(f"Available tokens for analysis completion: {available_tokens}")
        if available_tokens <= 0:
            self.loggers['llm'].error("Generate analysis: Not enough tokens available for completion.")
            raise ProcessingError("Prompt is too long to generate a valid completion.")
        max_completion_tokens = min(model_config['max_tokens'], available_tokens)
        try:
            response = await self.openai_client.chat.completions.create(
                model=model_config['model'],
                messages=messages,
                max_tokens=max_completion_tokens,
                temperature=model_config['temperature']
            )
            if not response.choices or not response.choices[0].message:
                self.loggers['llm'].error("OpenAI response is missing choices or messages.")
                raise ProcessingError("Failed to generate analysis due to invalid OpenAI response.")
            analysis = response.choices[0].message.content.strip()
            self.loggers['llm'].debug(f"Generated analysis for content: {content[:30]}... [truncated]")
            return analysis
        except Exception as e:
            self.loggers['llm'].error(f"Failed to generate analysis: {str(e)}")
            raise ProcessingError(f"Failed to generate analysis: {str(e)}")

    def truncate_text(self, text: str, max_chars: int) -> str:
        return text if len(text) <= max_chars else text[:max_chars] + "..."

    async def process_full_report(self, processed_results: List[ProcessingResult]) -> str:
        report_lines = ["# Refined Analysis Report", ""]
        unique_entities = {}
        for result in processed_results:
            if result.success and result.data:
                data = result.data
                story_id = data.get('story_id', 'Unknown')
                report_lines.append(f"## Story ID: {story_id}")
                entities = data.get('entities', [])
                for entity in entities:
                    wikidata_id = data.get('wiki_info', {}).get(entity['text'], {}).get('wikidata_id')
                    if wikidata_id and wikidata_id not in unique_entities:
                        unique_entities[wikidata_id] = {
                            "text": entity['text'],
                            "type": entity['type'],
                            "info": data.get('wiki_info', {}).get(entity['text'], {})
                        }
                if unique_entities:
                    report_lines.append("### Entities:")
                    report_lines.append("| Entity | Type | Wikidata ID |")
                    report_lines.append("|--------|------|-------------|")
                    for eid, details in unique_entities.items():
                        report_lines.append(f"| {details['text']} | {details['type']} | {eid} |")
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
        self.loggers['main'].info("Generated refined analysis report.")
        return report_content

# Story Processor Class
class StoryProcessor:
    def __init__(self, config_manager: ConfigManager, loggers: Dict[str, logging.Logger]):
        self.config_manager = config_manager
        self.loggers = loggers
        paths_config = self.config_manager.get_paths_config()
        self.stories_collection_name: str = ''
        self.reports_collection_name: str = ''
        self.vector_size: int = 1536
        self.qdrant_client: Optional[QdrantClient] = None
        self.stories: Dict[str, str] = {}
        self.analysis_results: Dict[str, Dict[str, Any]] = {}
        self.openai_client: Optional[AsyncOpenAI] = None
        self.report_post_processor: Optional[ReportPostProcessor] = None
        process_pool_config = self.config_manager.get_process_pool_config()
        max_workers = process_pool_config.get('max_workers', multiprocessing.cpu_count())
        self.process_pool = ProcessPoolExecutor(max_workers=max_workers)
        self.entity_processor: Optional[EnhancedEntityProcessor] = None

    def initialize(self):
        self._initialize_services()

    def _initialize_services(self) -> None:
        try:
            self._setup_openai()
            self._setup_qdrant()
            self._setup_entity_processor()
            self.loggers['main'].info("All services initialized successfully")
        except Exception as e:
            self.loggers['errors'].error(f"Service initialization failed: {str(e)}")
            raise

    def _setup_openai(self) -> None:
        load_dotenv()
        api_key = self.config_manager.get_openai_api_key()
        self.openai_client = AsyncOpenAI(api_key=api_key)
        self.report_post_processor = ReportPostProcessor(self.openai_client, self.config_manager, self.loggers)
        self.loggers['main'].info("OpenAI API key set and ReportPostProcessor initialized.")

    def _setup_qdrant(self) -> None:
        qdrant_config = self.config_manager.config.get('qdrant', {}).get('settings', {})
        try:
            self.qdrant_client = QdrantClient(
                host=qdrant_config.get('host', 'localhost'),
                port=int(qdrant_config.get('port', 6333)),
                timeout=self.config_manager.get_model_config(ModelType.EMBEDDING)['timeout']
            )
            self.loggers['main'].info("Qdrant client initialized successfully.")
            self.stories_collection_name = qdrant_config.get('stories_collection', {}).get('name', 'stories_collection')
            self.vector_size = int(qdrant_config.get('stories_collection', {}).get('vector_size', 1536))
            distance_metric_stories = qdrant_config.get('stories_collection', {}).get('distance', 'COSINE').upper()
            self._initialize_collection(self.stories_collection_name, self.vector_size, distance_metric_stories, 'stories')
            self.reports_collection_name = qdrant_config.get('reports_collection', {}).get('name', 'reports_collection')
            reports_vector_size = int(qdrant_config.get('reports_collection', {}).get('vector_size', 1536))
            distance_metric_reports = qdrant_config.get('reports_collection', {}).get('distance', 'COSINE').upper()
            self._initialize_collection(self.reports_collection_name, reports_vector_size, distance_metric_reports, 'reports')
        except Exception as e:
            self.loggers['errors'].error(f"Failed to initialize Qdrant client: {str(e)}")
            raise ProcessingError(f"Qdrant client initialization failed: {str(e)}")

    def _initialize_collection(self, collection_name: str, vector_size: int, distance_metric: str, collection_type: str) -> None:
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
                self.loggers['main'].info(f"Created Qdrant collection: {collection_name}")
            else:
                collection_info = self.qdrant_client.get_collection(collection_name)
                if collection_info.config.params.vectors.size != vector_size or \
                   collection_info.config.params.vectors.distance != models.Distance[distance_metric]:
                    raise ProcessingError(f"Existing collection '{collection_name}' has mismatched configuration")
                self.loggers['main'].info(f"Qdrant collection '{collection_name}' already exists with correct configuration.")
        except Exception as e:
            raise ProcessingError(f"Failed to initialize Qdrant collection '{collection_name}': {str(e)}")

    def _setup_entity_processor(self) -> None:
        if not self.openai_client or not self.config_manager:
            raise ProcessingError("OpenAI client or ConfigManager not initialized.")
        self.entity_processor = EnhancedEntityProcessor(self.openai_client, self.config_manager, self.loggers)
        self.loggers['main'].info("EnhancedEntityProcessor initialized.")

    @retry_async()
    async def _get_embedding_async(self, text: str) -> List[float]:
        try:
            model_config = self.config_manager.get_model_config(ModelType.EMBEDDING)
            response = await self.openai_client.embeddings.create(
                input=[text],
                model=model_config['model']
            )
            if not response.data or not response.data[0].embedding:
                self.loggers['llm'].error("OpenAI response is missing embedding data.")
                raise ProcessingError("Failed to generate embedding due to invalid OpenAI response.")
            embedding = response.data[0].embedding
            self.loggers['llm'].debug(f"Generated embedding for text: {text[:30]}... [truncated]. Embedding size: {len(embedding)}")
            return embedding
        except Exception as e:
            self.loggers['llm'].error(f"Failed to generate embedding: {str(e)}")
            raise ProcessingError(f"Failed to generate embedding: {str(e)}")

    def _prepare_qdrant_points(self, collection_name: str, point_id: str, content: str, wiki_info: Dict[str, Any], embedding: List[float], story_id: str) -> List[models.PointStruct]:
        if not isinstance(embedding, list) or len(embedding) != self.vector_size:
            raise ValueError(f"Invalid embedding: expected list of length {self.vector_size}")
        payload = {
            'report_id': point_id,
            'story_id': story_id,
            'content': content,
            'wiki_info': json.dumps(wiki_info),
            'timestamp': datetime.now().isoformat()
        }
        point = models.PointStruct(
            id=point_id,
            vector=embedding,
            payload=payload
        )
        self.loggers['main'].debug(f"Prepared Qdrant point for ID '{point_id}' in collection '{collection_name}'. Vector size: {len(embedding)}")
        return [point]

    def _generate_report_id(self) -> str:
        return str(uuid.uuid4())

    def _store_points_in_qdrant_sync(self, collection_name: str, points: List[models.PointStruct]) -> None:
        try:
            operation_info = self.qdrant_client.upsert(
                collection_name=collection_name,
                points=points
            )
            self.loggers['main'].info(f"Stored {len(points)} point(s) in Qdrant collection '{collection_name}'. Operation ID: {operation_info.operation_id}")
            self.loggers['main'].info(f"Upsert operation completed for collection '{collection_name}'")
        except UnexpectedResponse as e:
            raise ProcessingError(f"Failed to store points in Qdrant: {str(e)}")
        except Exception as e:
            raise ProcessingError(f"An unexpected error occurred while storing points in Qdrant: {str(e)}")

    async def _store_points_in_qdrant(self, collection_name: str, points: List[models.PointStruct]) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._store_points_in_qdrant_sync, collection_name, points)

    @retry_async()
    async def process_story_async(self, story_id: str, content: str) -> ProcessingResult:
        try:
            self.loggers['main'].debug(f"Starting preprocessing for story '{story_id}'")
            corrected_content = await asyncio.get_running_loop().run_in_executor(
                self.process_pool, preprocess_text_worker, content
            )
            self.loggers['main'].debug(f"Preprocessing completed for story '{story_id}'")
            resolved_entities = await asyncio.get_running_loop().run_in_executor(
                self.process_pool, extract_and_resolve_entities_worker, corrected_content
            )
            self.loggers['main'].debug(f"Entities extracted for story '{story_id}': {resolved_entities}")
            wiki_info = await self.entity_processor.get_entities_info(resolved_entities, corrected_content)
            self.loggers['llm'].debug(f"Generating embedding for story '{story_id}'")
            embedding = await self._get_embedding_async(corrected_content)
            self.loggers['llm'].debug(f"Embedding generated for story '{story_id}'")
            await self._save_intermediary_data_async(
                data=embedding,
                path=f"embeddings/{story_id}.json",
                data_type='embedding'
            )
            story_uuid = self._generate_report_id()
            points = self._prepare_qdrant_points(self.stories_collection_name, story_uuid, corrected_content, wiki_info, embedding, story_id)
            await self._store_points_in_qdrant(self.stories_collection_name, points)
            self.loggers['main'].debug(f"Stored embedding in Qdrant for story '{story_id}'")
            self.loggers['llm'].debug(f"Generating analysis for story '{story_id}'")
            analysis = await self.report_post_processor.generate_analysis(corrected_content, resolved_entities, wiki_info)
            self.loggers['llm'].debug(f"Analysis generated for story '{story_id}'")
            await self._save_intermediary_data_async(
                data=analysis,
                path=f"analysis/{story_id}.json",
                data_type='analysis'
            )
            chart_path = await asyncio.get_running_loop().run_in_executor(
                None, self.generate_entity_frequency_chart, resolved_entities, story_id
            )
            
            # Pass processed_result directly to refined report
            result = {
                'story_id': story_id,
                'entities': resolved_entities,
                'wiki_info': wiki_info,
                'analysis': analysis,  # Use analysis directly without refinement to ensure detailed report
                'embedding': embedding,
                'timestamp': datetime.now().isoformat(),
                'entity_frequency_chart': chart_path
            }
            self.loggers['main'].info(f"Successfully processed story '{story_id}'")
            return ProcessingResult(success=True, data=result)
        except ProcessingError as pe:
            self.loggers['errors'].error(f"Failed to process story '{story_id}': {str(pe)}")
            return ProcessingResult(success=False, error=str(pe))
        except Exception as e:
            self.loggers['errors'].error(f"Failed to process story '{story_id}': {str(e)}")
            return ProcessingResult(success=False, error=str(e))

    async def process_stories_async(self, stories_dir: str, output_path: str, summary_output_path: str) -> None:
        try:
            self._load_stories(stories_dir)
            concurrency_config = self.config_manager.get_concurrency_config()
            semaphore_limit = concurrency_config.get('semaphore_limit', multiprocessing.cpu_count() * 2)
            semaphore = asyncio.Semaphore(semaphore_limit)
            async def semaphore_wrapper(story_id, content):
                async with semaphore:
                    return await self.process_story_async(story_id, content)
            tasks = []
            for story_id, content in self.stories.items():
                self.loggers['main'].info(f"Processing story: {story_id}")
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
                    self.loggers['errors'].error(f"Error processing story: {str(result)}")
                    processed_results.append(ProcessingResult(success=False, error=str(result)))
                else:
                    self.loggers['errors'].error(f"Unexpected result type: {type(result)}")
                    processed_results.append(ProcessingResult(success=False, error="Unexpected result type"))
            
            # Generate refined report from processed_results
            refined_report = await self.report_post_processor.process_full_report(processed_results)
            await self._save_report_async(refined_report, output_path)
            
            # Store refined reports in Qdrant
            for result in processed_results:
                if result.success and result.data:
                    story_id = result.data['story_id']
                    report_content = result.data['analysis']
                    wiki_info_content = json.dumps(result.data['wiki_info'], indent=2)
                    report_id = self._generate_report_id()
                    embedding = await self._get_embedding_async(report_content)
                    report_points = self._prepare_qdrant_points(
                        self.reports_collection_name,
                        report_id,
                        report_content,
                        result.data['wiki_info'],
                        embedding,
                        story_id
                    )
                    await self._store_points_in_qdrant(self.reports_collection_name, report_points)
                    self.loggers['main'].info(f"Refined report for story '{story_id}' stored in '{self.reports_collection_name}' collection.")
            
            # Retrieve all reports for summary
            all_reports = await self._retrieve_all_reports_async(self.reports_collection_name)
            if not all_reports:
                self.loggers['main'].warning("No reports found in the reports collection to generate a summary.")
                summary = "No reports available to generate a summary."
            else:
                self.loggers['llm'].debug("Generating summary from all stored reports")
                summary = await self.report_post_processor.generate_summary(all_reports)
            await self._save_report_async(summary, summary_output_path)
            
            # Generate visualizations
            await asyncio.get_running_loop().run_in_executor(
                None, self.generate_word_cloud, processed_results, "word_cloud.png"
            )
            await asyncio.get_running_loop().run_in_executor(
                None, self.generate_entity_distribution, processed_results, "entity_distribution.html"  # Changed to HTML for Plotly
            )
            await asyncio.get_running_loop().run_in_executor(
                None, self.generate_embeddings_tsne, processed_results, "embeddings_tsne.png"
            )
            await asyncio.get_running_loop().run_in_executor(
                None, self.generate_story_length_histogram, processed_results, "story_length_histogram.png"
            )
            self.loggers['main'].info(f"Processing complete. Refined reports saved to '{self.reports_collection_name}' and summary saved to {summary_output_path}")
        except Exception as e:
            self.loggers['errors'].error(f"Failed to process stories: {str(e)}", exc_info=True)
        finally:
            if self.entity_processor:
                try:
                    await self.entity_processor.close()
                except Exception as e:
                    self.loggers['errors'].error(f"Failed to close entity processor session: {e}")
            if self.qdrant_client:
                try:
                    self.qdrant_client.close()
                    self.loggers['main'].info("Qdrant client connection closed.")
                except Exception as e:
                    self.loggers['errors'].error(f"Failed to close Qdrant client: {e}")
            if self.process_pool:
                try:
                    self.process_pool.shutdown(wait=True)
                    self.loggers['main'].info("Process pool shut down successfully.")
                except Exception as e:
                    self.loggers['errors'].error(f"Failed to shut down process pool: {e}")

    def _load_stories(self, stories_dir: str) -> None:
        stories_path = Path(stories_dir)
        if not stories_path.exists() or not stories_path.is_dir():
            raise ProcessingError(f"Stories directory not found: {stories_dir}")
        self.stories = {}
        for story_file in stories_path.glob('*.md'):
            try:
                with open(story_file, 'r', encoding='utf-8', errors='replace') as f:
                    content = f.read().strip()
                    if not content:
                        self.loggers['wikipedia'].warning(f"Story file '{story_file}' is empty. Skipping.")
                        continue
                    self.stories[story_file.stem] = content
                    self.loggers['main'].debug(f"Loaded story '{story_file.stem}'")
            except Exception as e:
                self.loggers['errors'].error(f"Failed to read story file '{story_file}': {str(e)}")
                continue
        if not self.stories:
            raise ProcessingError("No valid stories found to process.")
        self.loggers['main'].info(f"Loaded {len(self.stories)} story/stories for processing.")

    async def _save_report_async(self, report_content: str, output_path: str) -> None:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            async with aiofiles.open(output_file, 'w', encoding='utf-8', errors='replace') as f:
                await f.write(report_content)
            self.loggers['main'].info(f"Report saved to {output_path}")
        except Exception as e:
            self.loggers['errors'].error(f"Failed to save report to '{output_path}': {str(e)}")

    async def _save_intermediary_data_async(self, data: Any, path: str, data_type: str) -> None:
        intermediary_dir = self.config_manager.get_paths_config().get('intermediary_dir', 'output/intermediary')
        output_dir = Path(intermediary_dir) / path
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        try:
            if data_type in ['entities', 'wiki_info', 'embedding', 'analysis', 'additional_info']:
                async with aiofiles.open(output_dir, 'w', encoding='utf-8', errors='replace') as f:
                    await f.write(json.dumps(data, indent=2))
                self.loggers['main'].debug(f"Saved {data_type} for '{path}'")
            elif data_type == 'preprocessed_text':
                async with aiofiles.open(output_dir, 'w', encoding='utf-8', errors='replace') as f:
                    await f.write(data)
                self.loggers['main'].debug(f"Saved preprocessed text for '{path}'")
            else:
                self.loggers['main'].warning(f"Unknown data_type '{data_type}' for saving intermediary data.")
        except Exception as e:
            self.loggers['errors'].error(f"Failed to save intermediary data to '{path}': {str(e)}")

    def generate_entity_frequency_chart(self, entities: List[Dict[str, str]], story_id: str) -> Optional[str]:
        entity_counts = defaultdict(int)
        for entity in entities:
            entity_counts[entity['text']] += 1
        if not entity_counts:
            self.loggers['main'].warning(f"No entities found for story '{story_id}' to generate frequency chart.")
            return None
        entities_list = list(entity_counts.keys())
        counts = list(entity_counts.values())
        plt.figure(figsize=(10, 6))
        plt.barh(entities_list, counts, color='skyblue')
        plt.xlabel('Frequency')
        plt.title('Entity Frequency')
        plt.tight_layout()
        visualizations_dir = self.config_manager.get_paths_config().get('visualizations_dir', 'output/visualizations')
        chart_filename = f"entity_frequency_{story_id}.png"
        chart_path = Path(visualizations_dir) / chart_filename
        chart_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(chart_path)
        plt.close()
        self.loggers['main'].info(f"Entity frequency chart saved to {chart_path}")
        return str(chart_path)

    def generate_entity_category_distribution_chart(self, processed_results: List[ProcessingResult], output_path: str) -> None:
        entity_type_counts = defaultdict(int)
        category_counts = defaultdict(int)
        for result in processed_results:
            if result.success and result.data:
                wiki_info = result.data.get('wiki_info', {})
                for entity, info in wiki_info.items():
                    entity_type = info.get('type', 'OTHER').upper()
                    if entity_type in ['PERSON', 'ORG', 'GPE', 'EVENT', 'PRODUCT']:
                        entity_type_counts[entity_type] += 1
                    else:
                        entity_type_counts['OTHER'] += 1
                    categories = info.get('categories', [])
                    for category in categories:
                        category_counts[category] += 1
        if not entity_type_counts and not category_counts:
            self.loggers['main'].warning("No entity types or categories found for distribution chart.")
            return
        # Plot Entity Types
        if entity_type_counts:
            labels = list(entity_type_counts.keys())
            sizes = list(entity_type_counts.values())
            fig = px.pie(names=labels, values=sizes, title='Entity Type Distribution', hole=0.3)
            visualizations_dir = self.config_manager.get_paths_config().get('visualizations_dir', 'output/visualizations')
            chart_path = Path(visualizations_dir) / f"entity_type_distribution_{output_path}.html"
            chart_path.parent.mkdir(parents=True, exist_ok=True)
            fig.write_html(chart_path)
            self.loggers['main'].info(f"Interactive entity type distribution chart saved to {chart_path}")
        # Plot Top 10 Categories
        if category_counts:
            top_categories = dict(sorted(category_counts.items(), key=lambda item: item[1], reverse=True)[:10])
            categories = list(top_categories.keys())
            counts = list(top_categories.values())
            fig = px.bar(x=categories, y=counts, labels={'x': 'Wikipedia Categories', 'y': 'Frequency'}, title='Top 10 Wikipedia Categories for Entities')
            fig.update_layout(xaxis_tickangle=-45)
            chart_path = Path(visualizations_dir) / f"entity_category_distribution_{output_path}.html"
            fig.write_html(chart_path)
            self.loggers['main'].info(f"Interactive entity category distribution chart saved to {chart_path}")

    def generate_word_cloud(self, processed_results: List[ProcessingResult], output_path: str) -> None:
        entity_freq = defaultdict(int)
        for result in processed_results:
            if result.success and result.data:
                entities = result.data.get('entities', [])
                for entity in entities:
                    entity_freq[entity['text']] += 1
        if not entity_freq:
            self.loggers['main'].warning("No entities found for word cloud generation.")
            return
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(entity_freq)
        plt.figure(figsize=(15, 7.5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.tight_layout(pad=0)
        visualizations_dir = self.config_manager.get_paths_config().get('visualizations_dir', 'output/visualizations')
        chart_path = Path(visualizations_dir) / output_path
        chart_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(chart_path)
        plt.close()
        self.loggers['main'].info(f"Word cloud saved to {chart_path}")

    def generate_entity_distribution(self, processed_results: List[ProcessingResult], output_path: str) -> None:
        self.generate_entity_category_distribution_chart(processed_results, output_path)

    def generate_embeddings_tsne(self, processed_results: List[ProcessingResult], output_path: str) -> None:
        embeddings = []
        labels = []
        for result in processed_results:
            if result.success and result.data:
                embedding = result.data.get('embedding', [])
                if embedding:
                    embeddings.append(embedding)
                    labels.append(result.data.get('story_id', 'Unknown'))
        if not embeddings:
            self.loggers['main'].warning("No embeddings found for t-SNE visualization.")
            return
        try:
            embeddings_np = np.array(embeddings)  # Convert to NumPy array
            n_samples = embeddings_np.shape[0]
            if n_samples <= 1:
                self.loggers['main'].warning("Not enough samples to generate t-SNE plot.")
                return
            perplexity = 30  # Default perplexity
            if n_samples <= perplexity:
                perplexity = max(5, n_samples - 1)  # Adjust perplexity to be less than n_samples
                self.loggers['main'].info(f"Adjusted perplexity to {perplexity} based on number of samples: {n_samples}")
            tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, n_iter=300)
            embeddings_2d = tsne.fit_transform(embeddings_np)
            plt.figure(figsize=(12, 8))
            plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], s=50, alpha=0.7)
            for i, label in enumerate(labels):
                plt.annotate(label, (embeddings_2d[i, 0], embeddings_2d[i, 1]))
            plt.title('t-SNE Visualization of Story Embeddings')
            plt.xlabel('Dimension 1')
            plt.ylabel('Dimension 2')
            plt.tight_layout()
            visualizations_dir = self.config_manager.get_paths_config().get('visualizations_dir', 'output/visualizations')
            chart_path = Path(visualizations_dir) / output_path
            chart_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(chart_path)
            plt.close()
            self.loggers['main'].info(f"t-SNE embeddings plot saved to {chart_path}")
        except Exception as e:
            self.loggers['errors'].error(f"Failed to generate t-SNE plot: {str(e)}")

    def generate_story_length_histogram(self, processed_results: List[ProcessingResult], output_path: str) -> None:
        story_lengths = []
        for result in processed_results:
            if result.success and result.data:
                analysis = result.data.get('analysis', '')
                word_count = len(analysis.split())
                story_lengths.append(word_count)
        if not story_lengths:
            self.loggers['main'].warning("No story lengths found for histogram.")
            return
        plt.figure(figsize=(10, 6))
        plt.hist(story_lengths, bins=10, color='skyblue', edgecolor='black')
        plt.xlabel('Number of Words')
        plt.ylabel('Number of Stories')
        plt.title('Distribution of Story Lengths')
        plt.tight_layout()
        visualizations_dir = self.config_manager.get_paths_config().get('visualizations_dir', 'output/visualizations')
        chart_path = Path(visualizations_dir) / output_path
        chart_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(chart_path)
        plt.close()
        self.loggers['main'].info(f"Story length distribution histogram saved to {chart_path}")

    def _retrieve_all_reports_sync(self, collection_name: str) -> List[str]:
        try:
            all_reports = []
            limit = self.config_manager.config.get('qdrant', {}).get('settings', {}).get('retrieve_limit', 1000)
            scroll_result = self.qdrant_client.scroll(
                collection_name=collection_name,
                limit=limit,
                with_payload=True,
                with_vectors=False
            )
            if scroll_result is None:
                self.loggers['main'].warning(f"No data returned from '{collection_name}' collection.")
                return all_reports
            for batch in scroll_result:
                if not batch:
                    break
                for record in batch:
                    if isinstance(record, Record) and record.payload:
                        content = record.payload.get('content', '')
                        if content:
                            all_reports.append(content)
                    else:
                        self.loggers['errors'].warning(f"Unexpected record type or missing payload: {record}")
            self.loggers['main'].info(f"Retrieved {len(all_reports)} reports from '{collection_name}' collection.")
            return all_reports
        except Exception as e:
            self.loggers['errors'].error(f"Failed to retrieve reports from '{collection_name}' collection: {str(e)}")
            return []

    async def _retrieve_all_reports_async(self, collection_name: str) -> List[str]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._retrieve_all_reports_sync, collection_name)

    async def generate_summary_report(self, summary_output_path: str) -> None:
        all_reports = await self._retrieve_all_reports_async(self.reports_collection_name)
        if not all_reports:
            self.loggers['main'].warning("No reports found in the reports collection to generate a summary.")
            summary = "No reports available to generate a summary."
        else:
            self.loggers['llm'].debug("Generating summary from all stored reports")
            summary = await self.report_post_processor.generate_summary(all_reports)
        await self._save_report_async(summary, summary_output_path)

    def _generate_report(self, results: List[ProcessingResult]) -> str:
        # This method is deprecated in favor of process_full_report
        return ""

# Global workers
_worker_nlp = None
_worker_spell_checker = None

def initialize_worker():
    global _worker_nlp
    global _worker_spell_checker
    if _worker_nlp is None:
        _worker_nlp = spacy.load("en_core_web_sm")
    if _worker_spell_checker is None:
        _worker_spell_checker = SpellChecker()

def preprocess_text_worker(text: str) -> str:
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
                corrected_word = token.text
            corrected_tokens.append(corrected_word)
        else:
            token_text = token.text if token.text is not None else ""
            corrected_tokens.append(token_text)
    corrected_text = ' '.join(corrected_tokens)
    return corrected_text

def extract_and_resolve_entities_worker(text: str) -> List[Dict[str, str]]:
    global _worker_nlp
    if _worker_nlp is None:
        initialize_worker()
    relevant_entity_types = {'PERSON', 'ORG', 'GPE', 'EVENT', 'PRODUCT'}
    doc = _worker_nlp(text)
    entities = list(set([ent.text for ent in doc.ents if ent.label_ in relevant_entity_types]))
    entity_groups = defaultdict(list)
    for entity in entities:
        key = ''.join(e.lower() for e in entity if e.isalnum())
        entity_groups[key].append(entity)
    resolved_entities = []
    for group in entity_groups.values():
        primary_entity = max(group, key=len)
        doc_primary = _worker_nlp(primary_entity)
        if doc_primary.ents:
            entity_type = doc_primary.ents[0].label_
        else:
            entity_type = 'UNKNOWN'  # Set to 'UNKNOWN' if no entity type found
        resolved_entities.append({"text": primary_entity, "type": entity_type})
    return resolved_entities

# Main Execution Function
async def main():
    try:
        config_manager = ConfigManager()
        logger_factory = LoggerFactory(config_manager)
        loggers = logger_factory.loggers
        processor = StoryProcessor(config_manager, loggers)
        processor.initialize()
        paths_config = config_manager.get_paths_config()
        await processor.process_stories_async(
            stories_dir=paths_config.get('stories_dir', 'stories'),
            output_path=paths_config.get('output_report_path', 'output/report.md'),
            summary_output_path=paths_config.get('summary_output_path', 'output/summary_report.md')
        )
    except Exception as e:
        root_logger = logging.getLogger('errors')
        root_logger.error(f"Application error: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass
    asyncio.run(main())

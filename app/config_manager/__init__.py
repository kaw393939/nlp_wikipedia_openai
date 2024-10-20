# Config Manager Class
import logging
import os
from pathlib import Path
import threading
from typing import Any, Dict

import yaml

from app.exceptions import ConfigError
from app.mymodels import ModelType


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
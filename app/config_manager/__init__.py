# config_manager.py

import logging
import os
from pathlib import Path
import threading
from typing import Any, Dict, Optional

import yaml
from dotenv import load_dotenv

from app.exceptions import ConfigError
from app.mymodels import ModelType


class ConfigManager:
    """
    Singleton class to manage application configurations.
    It loads configurations from a YAML file and environment variables.
    """

    _instance: Optional['ConfigManager'] = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls, config_dir: str = "config", env_path: str = ".env"):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(ConfigManager, cls).__new__(cls)
                    cls._instance.config_dir = Path(config_dir)
                    cls._instance.config: Dict[str, Any] = {}
                    cls._instance._load_dotenv(env_path)
                    cls._instance._load_configs()
        return cls._instance

    def _load_dotenv(self, env_path: str) -> None:
        """
        Load environment variables from a .env file.
        """
        env_file = Path(env_path)
        if env_file.exists():
            load_dotenv(dotenv_path=env_file)
            logging.getLogger('main').info(f".env file loaded from {env_file.resolve()}")
        else:
            logging.getLogger('errors').warning(f".env file not found at {env_file.resolve()}")

    def _read_yaml_file(self, filepath: Path) -> Dict[str, Any]:
        """
        Read and parse a YAML configuration file.

        Args:
            filepath (Path): Path to the YAML file.

        Returns:
            Dict[str, Any]: Parsed configuration.

        Raises:
            ConfigError: If the file cannot be read or parsed.
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as file:
                content = file.read()
                # Substitute environment variables in the YAML file
                content = os.path.expandvars(content)
                return yaml.safe_load(content) or {}
        except FileNotFoundError:
            raise ConfigError(f"Configuration file not found: {filepath.resolve()}")
        except yaml.YAMLError as e:
            raise ConfigError(f"Error parsing YAML file {filepath.resolve()}: {str(e)}")
        except Exception as e:
            raise ConfigError(f"Unexpected error reading configuration file {filepath.resolve()}: {str(e)}")

    def _load_configs(self) -> None:
        """
        Load and validate the configuration from the YAML file.
        
        Raises:
            ConfigError: If required sections or keys are missing.
        """
        try:
            config_path = self.config_dir / 'config.yaml'
            if not config_path.exists():
                raise ConfigError(f"Configuration file does not exist: {config_path.resolve()}")

            self.config = self._read_yaml_file(config_path)
            logging.getLogger('main').debug(f"Raw Configurations Loaded: {self.config}")

            # Define required top-level sections
            required_sections = [
                'openai', 'qdrant', 'prompts', 'logging',
                'retry', 'aiohttp', 'process_pool', 'paths',
                'concurrency', 'analysis', 'cache'
            ]
            for section in required_sections:
                if section not in self.config:
                    raise ConfigError(f"Missing required configuration section: '{section}'")

            # Specific validation for OpenAI settings
            openai_settings = self.config.get('openai', {}).get('settings', {})
            if not openai_settings:
                raise ConfigError("Missing 'settings' section in 'openai' configuration")

            # Ensure OpenAI API key is provided via environment or config
            api_key = self.get_openai_api_key()
            if not api_key:
                raise ConfigError("OpenAI API key not found in environment variables or config.yaml")

            logging.getLogger('main').info("Configuration loaded and validated successfully")
        except ConfigError as e:
            logging.getLogger('errors').error(f"Configuration validation error: {str(e)}")
            raise
        except Exception as e:
            logging.getLogger('errors').error(f"Unexpected error during configuration loading: {str(e)}")
            raise

    def get_openai_api_key(self) -> str:
        """
        Retrieve the OpenAI API key from environment variables or configuration.

        Returns:
            str: OpenAI API key.

        Raises:
            ConfigError: If the API key is not found.
        """
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            logging.getLogger('main').debug("OPENAI_API_KEY retrieved from environment variables")
            return api_key

        # Fallback to config.yaml if environment variable is not set
        api_key = self.config.get('openai', {}).get('settings', {}).get('api_key')
        if api_key:
            logging.getLogger('main').warning("OPENAI_API_KEY retrieved from config.yaml. "
                                             "It is recommended to use environment variables for sensitive information.")
            return api_key

        raise ConfigError("OpenAI API key not found in environment variables or config.yaml")

    def get_model_config(self, model_type: ModelType) -> Dict[str, Any]:
        """
        Retrieve the configuration for a specific model type.

        Args:
            model_type (ModelType): The type of model (e.g., EMBEDDING, CHAT).

        Returns:
            Dict[str, Any]: Configuration for the specified model.

        Raises:
            ConfigError: If the model type is unknown or configuration is missing.
        """
        try:
            models_config = self.config['openai']['settings']
            if model_type == ModelType.EMBEDDING:
                return {
                    'model': models_config.get('embedding_model', 'text-embedding-ada-002'),
                    'timeout': float(models_config.get('timeout', 30.0)),
                    'context_length': int(models_config.get('context_length', 8192))
                }
            elif model_type == ModelType.CHAT:
                return {
                    'model': models_config.get('chat_model', 'gpt-4'),
                    'max_tokens': int(models_config.get('max_tokens', 4096)),
                    'temperature': float(models_config.get('temperature', 0.7)),
                    'context_length': int(models_config.get('context_length', 8192))
                }
            else:
                raise ValueError(f"Unknown model type: {model_type}")
        except KeyError as e:
            raise ConfigError(f"Missing configuration for {model_type}: {str(e)}")

    def get_prompt(self, prompt_type: str, role: str) -> str:
        """
        Retrieve a specific prompt based on its type and role.

        Args:
            prompt_type (str): The type of prompt (e.g., report_refinement).
            role (str): The role within the prompt (e.g., system, user).

        Returns:
            str: The requested prompt.

        Raises:
            ConfigError: If the prompt type or role is not found.
        """
        try:
            prompt = self.config['prompts'][prompt_type][role]
            if not prompt:
                raise ConfigError(f"Prompt not found for type '{prompt_type}' and role '{role}'")
            return prompt
        except KeyError as e:
            raise ConfigError(f"Missing prompt configuration: {str(e)}")

    def get_logging_config(self) -> Dict[str, Any]:
        """
        Retrieve the logging configuration.

        Returns:
            Dict[str, Any]: Logging configuration.
        """
        return self.config.get('logging', {})

    def get_retry_config(self) -> Dict[str, Any]:
        """
        Retrieve the retry configuration.

        Returns:
            Dict[str, Any]: Retry configuration.
        """
        return self.config.get('retry', {})

    def get_aiohttp_config(self) -> Dict[str, Any]:
        """
        Retrieve the aiohttp configuration.

        Returns:
            Dict[str, Any]: Aiohttp configuration.
        """
        return self.config.get('aiohttp', {})

    def get_process_pool_config(self) -> Dict[str, Any]:
        """
        Retrieve the process pool configuration.

        Returns:
            Dict[str, Any]: Process pool configuration.
        """
        return self.config.get('process_pool', {})

    def get_paths_config(self) -> Dict[str, Any]:
        """
        Retrieve the paths configuration.

        Returns:
            Dict[str, Any]: Paths configuration.
        """
        return self.config.get('paths', {})

    def get_concurrency_config(self) -> Dict[str, Any]:
        """
        Retrieve the concurrency configuration.

        Returns:
            Dict[str, Any]: Concurrency configuration.
        """
        return self.config.get('concurrency', {})

    def get_analysis_config(self) -> Dict[str, Any]:
        """
        Retrieve the analysis configuration.

        Returns:
            Dict[str, Any]: Analysis configuration.
        """
        return self.config.get('analysis', {})

    def get_qdrant_config(self) -> Dict[str, Any]:
        """
        Retrieve the Qdrant configuration.

        Returns:
            Dict[str, Any]: Qdrant configuration.
        """
        return self.config.get('qdrant', {})

    def get_cache_config(self) -> Dict[str, Any]:
        """
        Retrieve the cache configuration.

        Returns:
            Dict[str, Any]: Cache configuration.
        """
        return self.config.get('cache', {})

    def reload_configs(self) -> None:
        """
        Reload the configurations from the YAML file and environment variables.
        Useful for dynamic configuration updates.

        Raises:
            ConfigError: If reloading fails.
        """
        with self._lock:
            try:
                logging.getLogger('main').info("Reloading configurations...")
                self._load_dotenv(env_path=".env")  # Reload environment variables
                self._load_configs()  # Reload configurations
                logging.getLogger('main').info("Configurations reloaded successfully")
            except Exception as e:
                logging.getLogger('errors').error(f"Failed to reload configurations: {str(e)}")
                raise


# Example Usage
if __name__ == "__main__":
    # Configure root logger to output to console for demonstration purposes
    logging.basicConfig(level=logging.DEBUG,
                        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    try:
        config_manager = ConfigManager()
        openai_key = config_manager.get_openai_api_key()
        logging.info(f"OpenAI API Key Retrieved: {'***' + openai_key[-4:]}")  # Masked output
    except ConfigError as ce:
        logging.error(f"Configuration Error: {str(ce)}")
    except Exception as ex:
        logging.error(f"Unexpected Error: {str(ex)}")

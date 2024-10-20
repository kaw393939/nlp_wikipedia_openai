# Logger Factory Class
import logging
from pathlib import Path

from app.config_manager import ConfigManager


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

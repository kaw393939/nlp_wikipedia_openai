import logging
import asyncio
import multiprocessing
from app.config_manager import ConfigManager
from app.logging_setup import LoggerFactory
from app.story_processor import StoryProcessor

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

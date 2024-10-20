
# Retry Decorators
import asyncio
from functools import wraps
import logging
import time


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

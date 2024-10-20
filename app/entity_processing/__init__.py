# Enhanced Entity Processor Class
import asyncio
import logging
from typing import Any, Dict, List, Optional

import aiohttp
from openai import AsyncOpenAI

from app import retry_async
from app.config_manager import ConfigManager


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

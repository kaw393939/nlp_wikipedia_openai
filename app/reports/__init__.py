# app/reports.py

import asyncio
import json
import logging
import re
import warnings
from typing import Any, Dict, List, Optional

import spacy
import wikipedia
from aiolimiter import AsyncLimiter
from bs4 import GuessedAtParserWarning
from openai import AsyncOpenAI

from app.config_manager import ConfigManager
from app.exceptions import ProcessingError
from app.mymodels import ModelType
from app import retry_async  # Assuming you have a custom retry decorator

# Suppress the GuessedAtParserWarning from BeautifulSoup used within the wikipedia library
warnings.filterwarnings("ignore", category=GuessedAtParserWarning)


def sanitize_markdown(text: str) -> str:
    """
    Sanitize text to prevent Markdown formatting issues.

    Args:
        text (str): The text to sanitize.

    Returns:
        str: The sanitized text.
    """
    if not text:
        return ""
    # Escape pipe characters to prevent table breaks
    text = text.replace("|", "\\|")
    # Optionally, remove or replace other problematic characters
    return text.strip()


def validate_markdown_table(markdown: str, logger: logging.Logger) -> bool:
    """
    Validate that Markdown tables are correctly formatted.

    Args:
        markdown (str): The Markdown content to validate.
        logger (logging.Logger): Logger to record any validation errors.

    Returns:
        bool: True if all tables are valid, False otherwise.
    """
    table_pattern = re.compile(r'^\|.*\|$', re.MULTILINE)
    tables = table_pattern.findall(markdown)
    for table in tables:
        rows = table.strip().split('\n')
        if not rows:
            continue
        # Determine the number of columns from the header
        header = rows[0]
        num_cols = len(header.split('|')) - 2  # subtracting two for leading and trailing pipes
        # Check each row for consistent column count
        for row in rows:
            if (len(row.split('|')) - 2) != num_cols:
                logger.error(f"Malformed Markdown table detected: {row}")
                return False
    return True


class ReportPostProcessor:
    def __init__(
        self,
        openai_client: AsyncOpenAI,
        config_manager: ConfigManager,
        loggers: Dict[str, logging.Logger],
    ):
        """
        Initialize the ReportPostProcessor with necessary clients, configurations, and loggers.

        Args:
            openai_client (AsyncOpenAI): Asynchronous OpenAI client for LLM interactions.
            config_manager (ConfigManager): Manages configuration settings.
            loggers (Dict[str, logging.Logger]): Dictionary of loggers for different modules.
        """
        self.openai_client = openai_client
        self.config_manager = config_manager
        self.loggers = loggers
        self.limiter = AsyncLimiter(max_rate=5, time_period=1)  # Rate limiter for OpenAI API

        # Initialize SpaCy NLP model
        self.nlp = spacy.load("en_core_web_sm")

    @retry_async()
    async def generate_analysis(
        self,
        corrected_content: str,
        resolved_entities: List[Dict[str, Any]],
        wiki_info: Dict[str, Any],
        sentiment: Optional[str] = None,
        concepts: Optional[List[str]] = None,
        emotions: Optional[List[str]] = None,
        keywords: Optional[List[str]] = None,
        relations: Optional[Any] = None,
        summary: Optional[str] = None,
    ) -> str:
        """
        Generate a refined analysis report based on the provided data.

        Args:
            corrected_content (str): The preprocessed story content.
            resolved_entities (List[Dict[str, Any]]): List of resolved entities.
            wiki_info (Dict[str, Any]]): Retrieved Wikipedia information.
            sentiment (Optional[str]): Overall sentiment of the story.
            concepts (Optional[List[str]]): Key concepts extracted from the story.
            emotions (Optional[List[str]]): Emotions detected in the story.
            keywords (Optional[List[str]]): Keywords extracted from the story.
            relations (Optional[Any]): Relationships between entities.
            summary (Optional[str]): Summary of the story.

        Returns:
            str: The refined analysis report.
        """
        # Retrieve prompts from the configuration
        prompt_system = self.config_manager.get_prompt("report_refinement", "system")
        prompt_user_template = self.config_manager.get_prompt("report_refinement", "user")

        # Ensure that the prompt instructs the LLM to respond in English
        if "Respond in English" not in prompt_system:
            prompt_system += "\nRespond in English."

        # Prepare the data to fill in the prompt
        relations_str = json.dumps(relations) if relations else "No relations identified."
        prompt_user = prompt_user_template.format(
            original_story=corrected_content,
            generated_report=summary or "No summary available.",
            wikipedia_info=json.dumps(wiki_info, indent=2),
            sentiment=sentiment or "Not analyzed",
            concepts=', '.join(concepts) if concepts else "Not extracted",
            emotions=', '.join(emotions) if emotions else "Not detected",
            keywords=', '.join(keywords) if keywords else "Not extracted",
            relations=relations_str,
        )

        # Construct messages for OpenAI ChatCompletion
        messages = [
            {"role": "system", "content": prompt_system},
            {"role": "user", "content": prompt_user},
        ]

        self.loggers['llm'].debug(f"Messages sent to OpenAI for report refinement: {messages}")

        try:
            model_config = self.config_manager.get_model_config(ModelType.CHAT)
            self.loggers['llm'].debug(f"Model Configuration: {model_config}")

            async with self.limiter:
                response = await self.openai_client.chat.completions.create(
                    model=model_config['model'],
                    messages=messages,
                    max_tokens=model_config.get('max_tokens', 1500),
                    temperature=model_config.get('temperature', 0.7),
                    timeout=model_config.get('timeout', 60),
                )

            if response.choices and response.choices[0].message:
                refined_report = response.choices[0].message.content.strip()
                self.loggers['llm'].debug("Refined report generated successfully.")
                return refined_report
            else:
                self.loggers['llm'].error("OpenAI response is missing choices or messages.")
                raise ProcessingError("OpenAI returned an invalid response.")

        except Exception as e:
            self.loggers['errors'].error(f"Failed to generate analysis: {str(e)}")
            raise ProcessingError(f"Failed to generate analysis: {str(e)}")

    def correct_typographical_errors(self, entity_name: str) -> str:
        """
        Correct common typographical errors in entity names.

        Args:
            entity_name (str): The original entity name.

        Returns:
            str: The corrected entity name.
        """
        typo_corrections = {
            "MUON": "MUFON",
            "FEDERAL AVIATION ADMINISTRATION": "Federal Aviation Administration (FAA)",
            # Add more known typos and corrections as needed
        }

        corrected_name = typo_corrections.get(entity_name.upper(), entity_name)
        if corrected_name != entity_name:
            self.loggers['llm'].debug(f"Corrected '{entity_name}' to '{corrected_name}'")
        return corrected_name

    def extract_entities_nlp(self, text: str) -> List[Dict[str, str]]:
        """
        Extract entities using SpaCy's NER.

        Args:
            text (str): The text to analyze.

        Returns:
            List[Dict[str, str]]: A list of entities with their details.
        """
        entities = []
        doc = self.nlp(text)
        for ent in doc.ents:
            entities.append({
                "Entity": ent.text,
                "Type": ent.label_,
                "Description": "No description available."
            })
        self.loggers['llm'].debug(f"NLP extracted entities: {entities}")
        return entities

    def post_process_entities(self, entities: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Post-process the extracted entities to validate and enhance them.

        Args:
            entities (List[Dict[str, str]]): The list of extracted entities.

        Returns:
            List[Dict[str, str]]: The validated and enhanced list of entities.
        """
        processed_entities = []
        seen = set()
        for entity in entities:
            name = entity['Entity']
            if name.lower() in seen:
                continue
            seen.add(name.lower())
            # Correct typographical errors
            name = self.correct_typographical_errors(name)
            # Add Wikipedia links if available
            if not entity.get("Wikipedia Link"):
                wiki_search = self.search_wikipedia(name)
                if wiki_search:
                    entity['Wikipedia Link'] = wiki_search  # Store only the URL
            # Sanitize all fields to prevent Markdown issues
            entity['Entity'] = sanitize_markdown(entity['Entity'])
            entity['Type'] = sanitize_markdown(entity['Type'])
            entity['Description'] = sanitize_markdown(entity['Description'])
            if entity.get("Wikipedia Link"):
                # Ensure the link is properly formatted
                entity['Wikipedia Link'] = sanitize_markdown(f"[Wikipedia]({entity['Wikipedia Link']})")
            else:
                entity['Wikipedia Link'] = ""
            processed_entities.append(entity)
        self.loggers['llm'].debug(f"Post-processed entities: {processed_entities}")
        return processed_entities

    def search_wikipedia(self, entity_name: str) -> Optional[str]:
        """
        Search for the entity on Wikipedia and return the URL if found.

        Args:
            entity_name (str): The name of the entity.

        Returns:
            Optional[str]: The Wikipedia URL if found, else None.
        """
        try:
            page = wikipedia.page(entity_name)
            return page.url
        except wikipedia.exceptions.DisambiguationError as e:
            # Handle disambiguation by selecting the first option
            try:
                page = wikipedia.page(e.options[0])
                return page.url
            except Exception as inner_e:
                self.loggers['errors'].error(f"Disambiguation resolution failed for '{entity_name}': {inner_e}")
                return None
        except wikipedia.exceptions.PageError:
            self.loggers['llm'].debug(f"Wikipedia page not found for '{entity_name}'.")
            return None
        except Exception as e:
            self.loggers['errors'].error(f"Unexpected error during Wikipedia search for '{entity_name}': {e}")
            return None

    def extract_entities_from_analysis(self, analysis: str) -> List[Dict[str, str]]:
        """
        Extract entities from the '### Entities' and '### Missing Entities' sections in analysis.

        Args:
            analysis (str): The analysis text.

        Returns:
            List[Dict[str, str]]: A list of entities with their details.
        """
        entities = []
        try:
            self.loggers['llm'].debug("Starting entity extraction from analysis.")

            # Define sections to extract entities from
            sections = ['Entities', 'Missing Entities']
            for section_title in sections:
                self.loggers['llm'].debug(f"Extracting entities from section: {section_title}")

                # Regex to match the section content
                pattern = rf'###\s*{re.escape(section_title)}\s*\n+((?:.|\n)+?)(?=\n###|$)'
                match = re.search(
                    pattern,
                    analysis,
                    re.IGNORECASE
                )
                if not match:
                    self.loggers['errors'].error(f"{section_title} section not found in analysis.")
                    continue

                entity_section = match.group(1)
                self.loggers['llm'].debug(f"{section_title} section found: {entity_section}")

                if section_title == 'Entities':
                    # Extract table rows excluding the header and separator
                    table_rows = re.findall(r'^\|([^|]+)\|([^|]+)\|([^|]+)\|([^|]+)\|$', entity_section, re.MULTILINE)
                    self.loggers['llm'].debug(f"Table rows found in {section_title}: {table_rows}")

                    for row in table_rows:
                        entity_name = row[0].strip()
                        entity_type = row[1].strip()
                        description = row[2].strip()
                        wikipedia_link = row[3].strip()

                        # Skip separator rows or malformed rows
                        if re.match(r'^-+$', entity_name.replace(' ', '')):
                            self.loggers['llm'].debug("Skipping separator row.")
                            continue

                        # Extract URL from Wikipedia Link if it's a Markdown link
                        wiki_url_match = re.match(r'\[.*?\]\((https?://[^)]+)\)', wikipedia_link)
                        if wiki_url_match:
                            wiki_url = wiki_url_match.group(1)
                        else:
                            wiki_url = wikipedia_link  # Assume it's a direct URL or empty

                        # Clean description by removing any residual links or markdown
                        description_clean = re.sub(r'\[.*?\]\(https?://[^\)]+\)', '', description).strip()

                        entities.append({
                            "Entity": entity_name,
                            "Type": entity_type,
                            "Description": description_clean,
                            "Wikipedia Link": wiki_url
                        })

                elif section_title == 'Missing Entities':
                    # Assuming Missing Entities are listed as numbered lists
                    list_items = re.findall(r'\d+\.\s+\*\*(.*?)\*\*:\s+(.*)', entity_section)
                    self.loggers['llm'].debug(f"List items found in {section_title}: {list_items}")

                    for item in list_items:
                        entity_name = item[0].strip()
                        description = item[1].strip()

                        entities.append({
                            "Entity": entity_name,
                            "Type": "Missing",
                            "Description": description,
                            "Wikipedia Link": ""
                        })

            # Apply NLP-based extraction on the entire analysis
            nlp_entities = self.extract_entities_nlp(analysis)
            self.loggers['llm'].debug(f"NLP-based entities: {nlp_entities}")

            # Merge and filter entities to avoid duplicates
            for nlp_entity in nlp_entities:
                if not any(ent['Entity'].lower() == nlp_entity['Entity'].lower() for ent in entities):
                    entities.append(nlp_entity)

            # Post-process entities to validate and enrich
            processed_entities = self.post_process_entities(entities)

            self.loggers['llm'].debug(f"Final extracted entities after NLP integration: {processed_entities}")
            return processed_entities

        except Exception as e:
            self.loggers['errors'].error(f"Error extracting entities: {e}")
            return []

    async def process_full_report(self, processed_results: List[Any]) -> str:
        """
        Combine individual analyses into a full report.

        Args:
            processed_results (List[Any]): List of ProcessingResult objects.

        Returns:
            str: The full refined analysis report.
        """
        report_lines = ["# Refined Analysis Report\n"]
        all_analyses = []

        for result in processed_results:
            if getattr(result, 'success', False) and getattr(result, 'data', None):
                data = result.data
                story_id = getattr(data, 'story_id', 'Unknown')
                analysis = getattr(data, 'analysis', '')
                wiki_info = getattr(data, 'wiki_info', {})

                # Create story section
                story_section = [f"## Story ID: {story_id}\n"]

                if analysis:
                    # Create Analysis subsection
                    story_section.extend([
                        "### Analysis\n",
                        f"{analysis}\n\n",
                    ])

                    # Do not add '### Entities' and '### Missing Entities' sections here
                    all_analyses.append(analysis)
                else:
                    # No analysis available
                    story_section.extend([
                        "### Analysis\n",
                        "No analysis available for this story.\n",
                        "### Entities\n",
                        "No entities extracted for this story.\n",
                        "### Missing Entities\n",
                        "No missing entities identified.\n"
                    ])

                story_section.append("\n---\n")
                report_lines.extend(story_section)
            else:
                error_message = getattr(result, 'error', "Unknown error")
                error_message = sanitize_markdown(error_message)
                report_lines.extend([
                    f"## Story ID: Unknown\n",
                    f"Error processing story: {error_message}\n",
                    "\n---\n"
                ])

        # Generate summary from all analyses
        if all_analyses:
            try:
                overall_summary = await self.generate_summary(all_analyses)
                # Sanitize summary before adding
                overall_summary = sanitize_markdown(overall_summary)
                report_lines.extend([
                    "# Summary of All Reports\n",
                    f"{overall_summary}\n"
                ])
            except Exception as e:
                self.loggers['errors'].error(f"Failed to generate overall summary: {e}")
                report_lines.extend([
                    "# Summary of All Reports\n",
                    "Failed to generate summary due to an error.\n"
                ])
        else:
            report_lines.extend([
                "# Summary of All Reports\n",
                "No valid analyses available to generate a summary.\n"
            ])

        final_report = '\n'.join(report_lines)
        # Validate the final report's Markdown tables
        if not validate_markdown_table(final_report, self.loggers['errors']):
            self.loggers['errors'].error("Generated Markdown report contains malformed tables.")
            raise ProcessingError("Generated Markdown report contains malformed tables.")
        return final_report

    @retry_async()
    async def generate_summary(self, reports: List[str]) -> str:
        """
        Generate a summary from all reports.

        Args:
            reports (List[str]): List of all report contents.

        Returns:
            str: The generated summary.
        """
        if not reports or all(not report.strip() for report in reports):
            return "No valid analyses available to generate a summary."

        prompt_system = self.config_manager.get_prompt("summary_generation", "system")
        prompt_user_template = self.config_manager.get_prompt("summary_generation", "user")

        # Ensure that the prompt instructs the LLM to respond in English
        if "Respond in English" not in prompt_system:
            prompt_system += "\nRespond in English."

        prompt_user = prompt_user_template.format(reports='\n\n'.join(reports))

        messages = [
            {"role": "system", "content": prompt_system},
            {"role": "user", "content": prompt_user},
        ]

        self.loggers['llm'].debug("Generating summary from all stored reports.")

        try:
            model_config = self.config_manager.get_model_config(ModelType.CHAT)
            self.loggers['llm'].debug(f"Model Configuration for Summary: {model_config}")

            async with self.limiter:
                response = await self.openai_client.chat.completions.create(
                    model=model_config['model'],
                    messages=messages,
                    max_tokens=model_config.get('summary_max_tokens', 2000),
                    temperature=model_config.get('temperature', 0.7),
                    timeout=model_config.get('timeout', 60),
                )

            if response.choices and response.choices[0].message:
                summary = response.choices[0].message.content.strip()
                self.loggers['llm'].debug("Summary generated successfully.")
                return summary
            else:
                self.loggers['llm'].error("OpenAI response is missing choices or messages.")
                raise ProcessingError("OpenAI returned an invalid response.")

        except Exception as e:
            self.loggers['errors'].error(f"Failed to generate summary: {e}")
            raise ProcessingError(f"Failed to generate summary: {str(e)}")

    def get_entity_description(self, entity: Dict[str, str]) -> str:
        """
        Generate a description for an entity.

        Args:
            entity (Dict[str, str]): Entity information.

        Returns:
            str: Description of the entity.
        """
        description = entity.get("Description")
        if not description:
            wikidata_id = entity.get("Wikidata ID", "")
            if wikidata_id:
                description = f"Entity with Wikidata ID: {wikidata_id}"
            else:
                description = "No description available"
        return description

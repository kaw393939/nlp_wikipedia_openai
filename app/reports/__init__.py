import asyncio
import json
import logging
import re
from typing import Any, Dict, List, Optional

from aiolimiter import AsyncLimiter
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from openai import AsyncOpenAI

from app.config_manager import ConfigManager
from app.exceptions import ProcessingError
from app.mymodels import ModelType
from app import retry_async  # Assuming you have a custom retry decorator


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

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=10),
        retry=retry_if_exception_type((asyncio.TimeoutError, ProcessingError)),
        reraise=True,
    )
    async def refine_report(
        self,
        original_story: str,
        generated_report: str,
        wikipedia_info: str,
        wikipedia_sections: str,
    ) -> str:
        """
        Refine the generated report by integrating Wikipedia information and extracting entities.

        Args:
            original_story (str): The original story content.
            generated_report (str): The initial generated report.
            wikipedia_info (str): Aggregated Wikipedia information related to the story.
            wikipedia_sections (str): Specific sections from Wikipedia pages.

        Returns:
            str: The refined report with integrated information and populated entities.
        """
        prompt = self.config_manager.get_prompt("report_refinement", "user")
        if not prompt:
            raise ProcessingError("Report refinement prompt not found in configuration.")

        # Format the prompt with actual content
        formatted_prompt = prompt.format(
            original_story=original_story,
            generated_report=generated_report,
            wikipedia_info=wikipedia_info,
            wikipedia_sections=wikipedia_sections
        )

        # Log the final formatted prompt for debugging
        self.loggers['llm'].debug(f"Final report refinement prompt: {formatted_prompt}")

        messages = [
            {"role": "system", "content": self.config_manager.get_prompt("report_refinement", "system")},
            {"role": "user", "content": formatted_prompt},
        ]

        self.loggers['llm'].debug(f"Messages sent to OpenAI for report refinement: {json.dumps(messages)}")

        try:
            async with self.limiter:
                response = await self.openai_client.chat.completions.create(
                    model=self.config_manager.config['openai']['settings'].get("chat_model", "gpt-4o"),
                    messages=messages,
                    max_tokens=self.config_manager.config['openai']['settings'].get("max_tokens", 1000),
                    temperature=self.config_manager.config['openai']['settings'].get("temperature", 0.7),
                    timeout=self.config_manager.config['openai']['settings'].get("timeout", 60),
                )

            if not response.choices or not response.choices[0].message:
                self.loggers['llm'].error("OpenAI response is missing choices or messages.")
                raise ProcessingError("Failed to generate refined report due to invalid OpenAI response.")

            refined_report = response.choices[0].message.content.strip()
            self.loggers['llm'].debug(f"Refined report generated successfully.")

            # Validate and extract entities
            entities = self.extract_entities(refined_report)
            if not entities:
                self.loggers['errors'].error("Entities section not found or empty in the refined report.")
                raise ProcessingError("Entities extraction failed during report refinement.")

            return refined_report

        except (asyncio.TimeoutError, ProcessingError) as e:
            self.loggers['errors'].error(f"Report refinement failed: {str(e)}")
            raise

    def extract_entities(self, refined_report: str) -> List[Dict[str, str]]:
        """
        Extract entities from the refined report using regex to parse the Entities section.

        Args:
            refined_report (str): The refined report content.

        Returns:
            List[Dict[str, str]]: A list of entities with their details.
        """
        try:
            # Regex to find the Entities table
            match = re.search(
                r'### Entities:\n\| Entity\s*\|\s*Type\s*\|\s*Wikidata ID\s*\|\n\|[-\s|]+\|\n((?:\|.*\|\n)+)',
                refined_report,
                re.IGNORECASE
            )
            if not match:
                self.loggers['errors'].error("Entities table not found in the refined report.")
                return []

            table_content = match.group(1)
            lines = table_content.strip().split('\n')
            entities = []
            for line in lines:
                cols = [col.strip() for col in line.strip('|').split('|')]
                if len(cols) == 3:
                    entity = {
                        "Entity": cols[0],
                        "Type": cols[1],
                        "Wikidata ID": cols[2]
                    }
                    entities.append(entity)
            self.loggers['llm'].debug(f"Extracted entities: {entities}")
            return entities
        except Exception as e:
            self.loggers['errors'].error(f"Error extracting entities: {e}")
            return []

    @retry_async()
    async def generate_summary(self, reports: List[str]) -> str:
        """
        Generate a comprehensive summary from all refined reports.

        Args:
            reports (List[str]): List of refined report contents.

        Returns:
            str: The generated summary.
        """
        if not reports or all(not report.strip() for report in reports):
            return "No valid reports available to generate a summary."

        try:
            # Create a structured prompt with clear sections
            formatted_reports = "\n\n=== Next Report ===\n\n".join(
                f"Report {i+1}:\n{report}" for i, report in enumerate(reports) if report.strip()
            )

            prompt = f"""Analyze and synthesize the following reports into a comprehensive summary. 
            Focus on identifying common themes, patterns, and key insights across all reports.

            {formatted_reports}

            Please provide a detailed summary that:
            1. Identifies main themes and patterns across all reports
            2. Highlights key findings and their significance
            3. Draws meaningful connections between different reports
            4. Provides overall conclusions based on the collective analysis
            """

            messages = [
                {
                    "role": "system", 
                    "content": "You are an expert analyst specializing in synthesizing multiple reports into comprehensive summaries. Your summaries are detailed, well-structured, and highlight key patterns and insights."
                },
                {"role": "user", "content": prompt}
            ]

            self.loggers['llm'].debug("Generating summary from reports")

            async with self.limiter:
                response = await self.openai_client.chat.completions.create(
                    model=self.config_manager.config['openai']['settings'].get("chat_model", "gpt-4o"),
                    messages=messages,
                    max_tokens=self.config_manager.config['openai']['settings'].get("max_tokens", 2000),
                    temperature=0.7,
                    timeout=self.config_manager.config['openai']['settings'].get("timeout", 60)
                )

            if not response.choices or not response.choices[0].message:
                raise ProcessingError("Failed to generate summary due to invalid OpenAI response.")

            summary = response.choices[0].message.content.strip()
            self.loggers['llm'].debug("Summary generated successfully")
            return summary

        except Exception as e:
            self.loggers['errors'].error(f"Summary generation failed: {str(e)}")
            raise ProcessingError(f"Failed to generate summary: {str(e)}")

    async def process_full_report(self, processed_results: List[Any]) -> str:
        """
        Generate a full refined analysis report based on processed results.

        Args:
            processed_results (List[ProcessingResult]): A list of processed result objects.

        Returns:
            str: The complete refined analysis report in markdown format.
        """
        report_lines = ["# Refined Analysis Report\n"]
        all_analyses = []

        for result in processed_results:
            if result.success and result.data:
                data = result.data
                story_id = data.get('story_id', 'Unknown')
                analysis = data.get('analysis', '')  # Changed from 'refined_report' to 'analysis'
                entities = data.get('entities', [])
                wiki_info = data.get('wiki_info', {})

                # Create story section
                story_section = [f"## Story ID: {story_id}\n"]
                
                if analysis:
                    story_section.extend([
                        "### Analysis\n",
                        f"{analysis}\n\n",
                        "### Key Entities\n"
                    ])

                    # Add entity information
                    if entities:
                        story_section.append("| Entity | Type | Description |\n")
                        story_section.append("|--------|------|-------------|\n")
                        for entity in entities:
                            entity_name = entity.get('Entity', 'Unknown')
                            entity_type = entity.get('Type', 'Unknown')
                            entity_info = wiki_info.get(entity_name, {})
                            description = (entity_info.get('summary', 'No description available')[:100] + "...") if entity_info.get('summary') else 'No description available'
                            story_section.append(f"| {entity_name} | {entity_type} | {description} |\n")

                    all_analyses.append(analysis)
                else:
                    story_section.append("No analysis available for this story.\n")

                story_section.append("\n---\n")
                report_lines.extend(story_section)
            else:
                error_message = result.error or 'Unknown error'
                report_lines.extend([
                    f"## Story ID: Unknown\n",
                    f"Error processing story: {error_message}\n",
                    "\n---\n"
                ])

        # Generate summary from all analyses
        if all_analyses:
            try:
                summary = await self.generate_summary(all_analyses)
                report_lines.extend([
                    "# Summary of All Reports\n",
                    f"{summary}\n"
                ])
            except Exception as e:
                self.loggers['errors'].error(f"Failed to generate summary: {str(e)}")
                report_lines.extend([
                    "# Summary of All Reports\n",
                    "Failed to generate summary due to an error.\n"
                ])
        else:
            report_lines.extend([
                "# Summary of All Reports\n",
                "No valid analyses available to generate a summary.\n"
            ])

        return '\n'.join(report_lines)

    async def generate_analysis(self, story_content: str, entities: List[Dict[str, Any]], wiki_info: Dict[str, Any]) -> str:
        """
        Analyze the story content and generate a refined report based on the extracted entities.

        Args:
            story_content (str): The original story content.
            entities (List[Dict[str, Any]]): Extracted entities from the story.
            wiki_info (Dict[str, Any]): Wikipedia information related to the entities.

        Returns:
            str: The refined analysis report.
        """
        try:
            # Create a more structured prompt
            entity_details = []
            for entity in entities:
                entity_name = entity.get('Entity', 'Unknown')
                entity_info = wiki_info.get(entity_name, {})
                entity_details.append(f"""
Entity: {entity_name}
Type: {entity.get('Type', 'Unknown')}
Summary: {entity_info.get('summary', 'No summary available')}
Additional Information: {json.dumps(entity_info.get('sections', {}), indent=2)}
""")

            prompt = f"""Provide a comprehensive analysis of the following story:

Story Content:
{story_content}

Key Entities and Their Information:
{"".join(entity_details)}

Please analyze this information and provide:
1. Main themes and key points of the story
2. Detailed analysis of each significant entity and their role
3. Integration of Wikipedia context to provide deeper insights
4. Identification of patterns and connections
5. Overall significance and implications

Format your response in clear sections with markdown headers."""

            messages = [
                {
                    "role": "system", 
                    "content": "You are an expert analyst specializing in story analysis and contextual interpretation. Provide detailed, well-structured analyses that incorporate all available information."
                },
                {"role": "user", "content": prompt}
            ]

            self.loggers['llm'].debug(f"Generating analysis for story")

            async with self.limiter:
                response = await self.openai_client.chat.completions.create(
                    model=self.config_manager.config['openai']['settings'].get("chat_model", "gpt-4o"),
                    messages=messages,
                    max_tokens=self.config_manager.config['openai']['settings'].get("max_tokens", 1000),
                    temperature=0.7,
                    timeout=self.config_manager.config['openai']['settings'].get("timeout", 60)
                )

            if not response.choices or not response.choices[0].message:
                raise ProcessingError("OpenAI response is missing choices or messages.")

            analysis = response.choices[0].message.content.strip()
            self.loggers['llm'].debug("Analysis generated successfully")
            return analysis

        except Exception as e:
            self.loggers['errors'].error(f"Failed to generate analysis: {str(e)}")
            raise ProcessingError(f"Failed to generate analysis: {str(e)}")

    def get_entity_description(self, entity: Dict[str, str]) -> str:
        """
        Generate a description for an entity.

        Args:
            entity (Dict[str, str]): Entity information.

        Returns:
            str: Description of the entity.
        """
        description = entity.get("description")
        if not description:
            wikidata_id = entity.get("Wikidata ID", "")
            if wikidata_id:
                description = f"Entity with Wikidata ID: {wikidata_id}"
            else:
                description = "No description available"
        return description
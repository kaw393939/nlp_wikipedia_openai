import json
import logging
from typing import Any, Dict, List

from openai import AsyncOpenAI
import tiktoken

from app import retry_async
from app.config_manager import ConfigManager
from app.exceptions import ProcessingError
from app.mymodels import ModelType
from app.results_processing.___init___ import ProcessingResult


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
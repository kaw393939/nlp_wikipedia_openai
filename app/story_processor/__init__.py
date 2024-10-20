# Story Processor Class
import asyncio
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
import json
import logging
import multiprocessing
from pathlib import Path
from typing import Any, Dict, List, Optional
import uuid
import aiofiles
from dotenv import load_dotenv
from matplotlib import pyplot as plt
import numpy as np
from openai import AsyncOpenAI
from qdrant_client import QdrantClient
from sklearn.manifold import TSNE
from wordcloud import WordCloud
from app import retry_async
from app.config_manager import ConfigManager
from app.entity_processing import EnhancedEntityProcessor
from app.exceptions import ProcessingError
from app.reports import ReportPostProcessor
from app.results_processing.___init___ import ProcessingResult
from qdrant_client.http.exceptions import UnexpectedResponse
import plotly.express as px
from app.workers import extract_and_resolve_entities_worker, preprocess_text_worker
from qdrant_client import models
from app.mymodels import ModelType


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
                timeout=self.config_manager.get_model_config(ModelType.EMBEDDING)['timeout']  # Using the ModelType from your own module
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
            
            result = {
                'story_id': story_id,
                'entities': resolved_entities,
                'wiki_info': wiki_info,
                'analysis': analysis,
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
                None, self.generate_entity_distribution, processed_results, "entity_distribution.html"
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
                    if isinstance(record, models.Record) and record.payload:
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

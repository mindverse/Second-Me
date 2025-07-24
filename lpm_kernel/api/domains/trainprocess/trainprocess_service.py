import gc
import json
import os
import re
import subprocess
import threading
import time
from typing import Optional, Dict

import psutil
from flask import jsonify

from lpm_kernel.L2.utils import save_hf_model
from lpm_kernel.api.common.responses import APIResponse
from lpm_kernel.api.common.script_executor import ScriptExecutor
from lpm_kernel.api.domains.loads.services import LoadService
from lpm_kernel.api.domains.trainprocess.L1_exposure_manager import output_files, query_l1_version_data, \
    read_file_content
from lpm_kernel.api.domains.trainprocess.process_step import ProcessStep
from lpm_kernel.api.domains.trainprocess.progress_enum import Status
from lpm_kernel.api.domains.trainprocess.progress_holder import TrainProgressHolder
from lpm_kernel.api.domains.trainprocess.training_params_manager import TrainingParamsManager
from lpm_kernel.base.convert_data import convert_standard_data
from lpm_kernel.base.database_operate import store_version, store_bio
from lpm_kernel.common.repository.database_session import DatabaseSession
from lpm_kernel.configs.config import Config
from lpm_kernel.configs.logging import get_train_process_logger, TRAIN_LOG_FILE
from lpm_kernel.file_data.chunker import DocumentChunker
from lpm_kernel.file_data.document_repository import DocumentRepository
from lpm_kernel.file_data.document_service import document_service
from lpm_kernel.kernel.chunk_service import ChunkService
from lpm_kernel.models.l1 import L1Version
from lpm_kernel.models.memory import Memory
from lpm_kernel.stage1.data import Stage1Data
from lpm_kernel.stage2.bio_qa import BioQAData
from lpm_kernel.stage2.global_bio.base import GlobalBioV2
from lpm_kernel.stage2.memqa.description import DescriptionData
from lpm_kernel.file_data.document_service import document_service
from lpm_kernel.stage2.memqa.diversity import DiversityData
from lpm_kernel.stage2.memqa.entity import EntityData
from lpm_kernel.stage2.memqa.relation import RelationshipData
from lpm_kernel.stage2.shades.shades_generator import ShadeGenerate, ShadeContentGenerate
from lpm_kernel.stage2.topics.topics_generator import TopicGenerate
from lpm_kernel.stage2.wiki.wiki_generator import wiki_gen
from lpm_kernel.stage3.generate_no_notes_questions import NoNoteDataGenerator
from lpm_kernel.stage3.generate_notes_questions import SyntheticDataGenerator

logger = get_train_process_logger()


class TrainProcessService:
    """Training process service (singleton pattern)"""

    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, current_model_name: str):
        if current_model_name is None:
            raise ValueError("current_model_name cannot be None")

        if not self._initialized:
            # Generate a unique progress file name based on model name
            self.progress = TrainProgressHolder(current_model_name)
            self.model_name = current_model_name  # Set model name directly
            self._initialized = True

            # Initialize stop flag
            self.is_stopped = False
            self.current_step = None

            params_manager = TrainingParamsManager()
            training_params = params_manager.get_latest_training_params()
            self.language = training_params["language"]

            # Data filtering parameters will be retrieved from TrainingParamsManager when needed

        # Update model name and progress instance if model name changes
        if current_model_name != self.model_name:
            self.model_name = current_model_name
            # Create new progress instance with updated progress file name
            self.progress = TrainProgressHolder(current_model_name)

    @classmethod
    def get_instance(cls, current_model_name: str = None):
        """Get the current instance of TrainProcessService
        
        Args:
            current_model_name: Optional model name to update the instance with
            
        Returns:
            TrainProcessService: The singleton instance
        """
        if cls._instance is None:
            if current_model_name is None:
                logger.warning("current_model_name must be provided when creating a new instance")
                return None
            return cls(current_model_name)

        if current_model_name is not None:
            # Update the existing instance with new model name
            cls._instance.model_name = current_model_name
            cls._instance.progress = TrainProgressHolder(current_model_name)

        return cls._instance

    def list_documents(self):
        """List all documents"""
        try:
            # Mark step as in progress
            self.progress.mark_step_status(ProcessStep.LIST_DOCUMENTS, Status.IN_PROGRESS)
            # Directly call document service instead of API
            documents = document_service.list_documents()
            # Mark step as completed if we found documents
            self.progress.mark_step_status(ProcessStep.LIST_DOCUMENTS, Status.COMPLETED)

            return [doc.to_dict() for doc in documents]
        except Exception as e:
            logger.error(f"List documents failed: {str(e)}")
            self.progress.mark_step_status(ProcessStep.LIST_DOCUMENTS, Status.FAILED)
            return []

    def generate_document_embeddings(self) -> bool:
        """Process embeddings for all documents"""
        try:
            # Mark step as in progress
            self.progress.mark_step_status(ProcessStep.GENERATE_DOCUMENT_EMBEDDINGS, Status.IN_PROGRESS)

            unembedding_docs = document_service._repository.find_unembedding()
            logger.info(f"Found {len(unembedding_docs)} documents that need embedding generation")

            if not unembedding_docs:
                logger.info("No documents need embedding generation, marking step as completed")
                self.progress.mark_step_status(ProcessStep.GENERATE_DOCUMENT_EMBEDDINGS, Status.COMPLETED)
                return True

            for doc in unembedding_docs:
                doc_id = doc.id

                # Directly call document service instead of API
                embedding = document_service.process_document_embedding(doc_id)
                if embedding is None:
                    logger.error(
                        f"Generate document embeddings failed for doc_id: {doc_id}"
                    )
                    self.progress.mark_step_status(ProcessStep.GENERATE_DOCUMENT_EMBEDDINGS, Status.FAILED)
                    return False
                logger.info(f"Successfully generated embedding for document {doc_id}")

            self.progress.mark_step_status(ProcessStep.GENERATE_DOCUMENT_EMBEDDINGS, Status.COMPLETED)
            return True
        except Exception as e:
            logger.error(f"Generate document embeddings failed: {str(e)}")
            self.progress.mark_step_status(ProcessStep.GENERATE_DOCUMENT_EMBEDDINGS, Status.FAILED)
            return False

    def process_chunks(self) -> bool:
        """Process document chunks"""
        try:
            # Mark step as in progress
            self.progress.mark_step_status(ProcessStep.CHUNK_DOCUMENT, Status.IN_PROGRESS)
            config = Config.from_env()
            chunker = DocumentChunker(
                chunk_size=int(config.get("DOCUMENT_CHUNK_SIZE")),
                overlap=int(config.get("DOCUMENT_CHUNK_OVERLAP")),
            )
            documents = document_service.list_documents()
            processed, failed, skipped = 0, 0, 0

            chunk_service = ChunkService()
            for doc in documents:
                try:
                    existing_chunks = document_service._repository.find_chunks(doc.id)
                    if existing_chunks and len(existing_chunks) > 0:
                        logger.info(f"Document {doc.id} already has {len(existing_chunks)} chunks, skipping...")
                        skipped += 1
                        continue

                    if not doc.raw_content:
                        logger.warning(f"Document {doc.id} has no content, skipping...")
                        failed += 1
                        continue

                    # Split into chunks and save
                    chunks = chunker.split(doc.raw_content)
                    for chunk in chunks:
                        chunk.document_id = doc.id
                        chunk_service.save_chunk(chunk)

                    processed += 1
                    logger.info(
                        f"Document {doc.id} processed: {len(chunks)} chunks created"
                    )
                except Exception as e:
                    logger.error(f"Failed to process document {doc.id}: {str(e)}")
                    failed += 1

            logger.info(f"Chunk processing completed: {processed} processed, {skipped} skipped, {failed} failed")
            self.progress.mark_step_status(ProcessStep.CHUNK_DOCUMENT, Status.COMPLETED)
            return True
        except Exception as e:
            logger.error(f"Process chunks failed: {str(e)}")
            self.progress.mark_step_status(ProcessStep.CHUNK_DOCUMENT, Status.FAILED)
            return False

    def chunk_embedding(self) -> bool:
        """Process embeddings for all document chunks"""
        try:
            # Mark step as in progress
            self.progress.mark_step_status(ProcessStep.CHUNK_EMBEDDING, Status.IN_PROGRESS)
            documents = self.list_documents()
            for doc in documents:
                doc_id = doc.get("id")
                try:
                    # Directly call document service to generate chunk embeddings
                    processed_chunks = document_service.generate_document_chunk_embeddings(doc_id)
                    if not processed_chunks:
                        logger.warning(f"No chunks to process for document: {doc_id}")
                        continue
                except Exception as e:
                    logger.error(
                        f"Generate chunk embeddings failed for doc_id: {doc_id}: {str(e)}"
                    )
                    self.progress.mark_step_status(ProcessStep.CHUNK_EMBEDDING, Status.FAILED)
                    return False
            # All documents' chunks processed successfully
            self.progress.mark_step_status(ProcessStep.CHUNK_EMBEDDING, Status.COMPLETED)
            return True
        except Exception as e:
            logger.error(f"Generate chunk embeddings failed: {str(e)}")
            self.progress.mark_step_status(ProcessStep.CHUNK_EMBEDDING, Status.FAILED)
            return False

    def extract_dimensional_topics(self) -> bool:
        """Extract dimensional topics (L0)"""
        try:
            # Mark step as in progress
            self.progress.mark_step_status(ProcessStep.EXTRACT_DIMENSIONAL_TOPICS, Status.IN_PROGRESS)
            logger.info("Starting dimensional topics extraction (L0)...")

            # Generate L0 - Call document_service to analyze all documents
            logger.info("Generating topics...")
            topic_generate = TopicGenerate()
            topics = []
            preferredLanguage = self.language
            result = topic_generate.topics_generate(topics=topics, preferredLanguage=preferredLanguage)

            logger.info(f"Successfully generated {len(result)} topics")
            analyzed_docs = document_service.analyze_all_documents()
            logger.info(f"Successfully analyzed {len(analyzed_docs)} documents")
            # Mark step as completed
            self.progress.mark_step_status(ProcessStep.EXTRACT_DIMENSIONAL_TOPICS, Status.COMPLETED)
            logger.info("Dimensional topics extraction (L0) completed successfully")
            return True

        except Exception as e:
            logger.error(f"Extract dimensional topics (L0) failed: {str(e)}")
            self.progress.mark_step_status(ProcessStep.EXTRACT_DIMENSIONAL_TOPICS, Status.FAILED)
            return False

    def generate_shades(self):
        try:
            # Mark step as in progress
            self.progress.mark_step_status(ProcessStep.GENERATE_SHADES, Status.IN_PROGRESS)
            logger.info("Starting shades generation...")

            logger.info("Generating shades...")

            doc_repository = DocumentRepository()
            documents = doc_repository.list()

            topics_path = "resources/data/stage2/topics/topic.json"
            with open(topics_path, "r") as f:
                topics_data = json.load(f)
                topics = topics_data.get("topics", [])

            # 测试shade generate
            shade_generate = ShadeGenerate()
            shades = []
            preferredLanguage = self.language
            shades_result = shade_generate.shades_generate(topics=topics, shades=shades,
                                                           preferredLanguage=preferredLanguage)
            logger.info(f"shades generate result: {shades_result}")
            # 保存shades result
            shades_result_json = json.dumps(shades_result, ensure_ascii=False, indent=2)
            os.makedirs("resources/data/stage2/shades", exist_ok=True)
            with open("resources/data/stage2/shades/shades.json", "w", encoding="utf-8") as f:
                f.write(shades_result_json)

            with open("resources/data/stage2/shades/shades.json", "r", encoding="utf-8") as f:
                shades_result = json.load(f)
            final_result = shades_result
            shade_content_generate = ShadeContentGenerate()
            for idx, shade in enumerate(shades_result["shades"]):

                cur_note = []
                cur_topic = []

                for note in documents:
                    if note.id in shade["relatedNotes"]:
                        cur_note.append(note)
                for topic in topics:
                    if topic["topicName"] in shade["sourceTopics"]:
                        cur_topic.append(topic)
                shades_content_input = {
                    "noteMemory": cur_note,
                    "topics": cur_topic,
                    "shades": [shade],
                    "preferredLanguage": self.language
                }
                shades_content_result = shade_content_generate._call(shades_content_input)
                logger.info(f"shade content generate result: {shades_content_result}")
                final_result["shades"][idx]["shadeContent"] = shades_content_result["shades"][0]["shadeContent"]
                final_result["shades"][idx]["shadeContentThirdView"] = shades_content_result["shades"][0][
                    "shadeContentThirdView"]
            # 保存shades content result
            with open("resources/data/stage2/shades/shades_content.json", "w", encoding="utf-8") as f:
                json.dump(final_result, f, indent=4, ensure_ascii=False)

            logger.info("Successfully generated shades")

            # Mark step as completed
            self.progress.mark_step_status(ProcessStep.GENERATE_SHADES, Status.COMPLETED)
            logger.info("Shades generation completed successfully")
            return True

        except Exception as e:
            logger.error(f"Generate shades failed: {str(e)}")
            self.progress.mark_step_status(ProcessStep.GENERATE_SHADES, Status.FAILED)
            return False

    def generate_biography(self) -> bool:
        """Generate biography using stage2 data"""
        try:
            # Mark step as in progress
            self.progress.mark_step_status(ProcessStep.GENERATE_BIOGRAPHY, Status.IN_PROGRESS)
            logger.info("Starting biography generation...")

            # Generate stage2 data and biography
            logger.info("Generating stage2 data and biography...")
            global_bio_v2 = GlobalBioV2()
            with open("resources/data/stage2/shades/shades_content.json", "r") as f:
                shades = json.load(f)
            shades = shades["shades"]

            inputs = {
                "oldGlobalBio": {
                    "content": "",
                    "summary": "",
                    "shadesList": shades
                },
                "preferredLanguage": self.language
            }
            result = global_bio_v2._call(inputs)
            with DatabaseSession.session() as session:
                new_version = session.query(L1Version).order_by(
                    L1Version.version.desc()).first().version + 1 if session.query(L1Version).order_by(
                    L1Version.version.desc()).first() else 1
                store_version(session, new_version)
                store_bio(session, new_version, result["globalBio"])
            os.makedirs("resources/data/stage2/global_bio", exist_ok=True)
            with open("resources/data/stage2/global_bio/global_bio.json", "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False)
            logger.info("Successfully generated stage2 data and biography")

            self.progress.mark_step_status(ProcessStep.GENERATE_BIOGRAPHY, Status.COMPLETED)
            logger.info("Biography generation completed successfully")
            return True

        except Exception as e:
            logger.error(f"Biography generation failed: {str(e)}")
            self.progress.mark_step_status(ProcessStep.GENERATE_BIOGRAPHY, Status.FAILED)
            return False

    def model_download(self) -> bool:
        """Download model"""
        try:
            # Mark step as in progress
            self.progress.mark_step_status(ProcessStep.MODEL_DOWNLOAD, Status.IN_PROGRESS)
            # Directly call save_hf_model function to download model
            logger.info(f"Starting model download: {self.model_name}")

            # Start monitoring the download progress in a separate thread
            monitor_thread = threading.Thread(target=self._monitor_model_download)
            monitor_thread.daemon = True
            monitor_thread.start()

            # Start the actual download
            model_path = save_hf_model(self.model_name)

            if model_path and os.path.exists(model_path):
                logger.info(f"Model downloaded successfully to {model_path}")
                self.progress.mark_step_status(ProcessStep.MODEL_DOWNLOAD, Status.COMPLETED)
                return True
            else:
                logger.error(f"Model path does not exist after download: {model_path}")
                self.progress.mark_step_status(ProcessStep.MODEL_DOWNLOAD, Status.FAILED)
                return False

        except Exception as e:
            logger.error(f"Download model failed: {str(e)}")
            self.progress.mark_step_status(ProcessStep.MODEL_DOWNLOAD, Status.FAILED)
            return False

    def generate_base(self) -> bool:
        """Stage1"""
        try:
            # Mark step as in progress
            self.progress.mark_step_status(ProcessStep.GENERATE_BASE, Status.IN_PROGRESS)
            logger.info("Generating base data...")
            data = Stage1Data()
            data.run()

            self.progress.mark_step_status(ProcessStep.GENERATE_BASE, Status.COMPLETED)
            logger.info("Base data generation completed successfully")
            return True
        except Exception as e:
            logger.error(f"Base data generation failed: {str(e)}")
            self.progress.mark_step_status(ProcessStep.GENERATE_BASE, Status.FAILED)
            return False

    def bio_qa_generation(self) -> bool:
        """Generate bio_qa using stage2 data"""
        try:
            # Mark step as in progress
            self.progress.mark_step_status(ProcessStep.BIO_QA_GENERATION, Status.IN_PROGRESS)
            logger.info("Starting bio_qa generation...")

            # Generate stage2 data and bio_qa
            logger.info("Generating stage2 data and bio_qa...")
            bio_qa = BioQAData()
            bio_qa.run()
            logger.info("Successfully generated stage2 data and bio_qa")

            self.progress.mark_step_status(ProcessStep.BIO_QA_GENERATION, Status.COMPLETED)
            logger.info("Bio_qa generation completed successfully")
            return True

        except Exception as e:
            logger.error(f"Decode preference patterns failed: {str(e)}")
            self.progress.mark_step_status(ProcessStep.BIO_QA_GENERATION, Status.FAILED)
            return False

    def wiki_data_generation(self) -> bool:
        """Generate wiki data"""
        try:
            # Mark step as in progress
            self.progress.mark_step_status(ProcessStep.WIKI_DATA_GENERATION, Status.IN_PROGRESS)
            logger.info("Starting wiki data generation...")

            # Generate wiki data
            logger.info("Generating wiki data...")
            params_manager = TrainingParamsManager()
            training_params = params_manager.get_latest_training_params()
            wiki_gen(preferred_language=training_params.get("language"))
            logger.info("Successfully generated wiki data")

            self.progress.mark_step_status(ProcessStep.WIKI_DATA_GENERATION, Status.COMPLETED)
            logger.info("Wiki data generation completed successfully")
            return True

        except Exception as e:
            logger.error(f"Generate wiki data failed: {str(e)}")
            self.progress.mark_step_status(ProcessStep.WIKI_DATA_GENERATION, Status.FAILED)
            return False

    def generate_memqa_entity(self) -> bool:
        """Generate memqa entity"""
        try:
            # Mark step as in progress
            self.progress.mark_step_status(ProcessStep.GENERATE_MEMQA_ENTITY, Status.IN_PROGRESS)
            logger.info("Starting memqa entity generation...")

            logger.info("Generating memqa entity...")
            entity = EntityData()
            entity.run()
            logger.info("Successfully generated memqa entity")

            self.progress.mark_step_status(ProcessStep.GENERATE_MEMQA_ENTITY, Status.COMPLETED)

            return True

        except Exception as e:
            logger.error(f"Generate memqa entity failed: {str(e)}")
            self.progress.mark_step_status(ProcessStep.GENERATE_MEMQA_ENTITY, Status.FAILED)
            return False

    def generate_memqa_relation(self) -> bool:
        """Generate memqa relation"""
        try:
            # Mark step as in progress
            self.progress.mark_step_status(ProcessStep.GENERATE_MEMQA_RELATION, Status.IN_PROGRESS)
            logger.info("Starting memqa relation generation...")

            logger.info("Generating memqa relation...")
            relation = RelationshipData()
            relation.run()
            logger.info("Successfully generated memqa relation")

            self.progress.mark_step_status(ProcessStep.GENERATE_MEMQA_RELATION, Status.COMPLETED)

            return True

        except Exception as e:
            logger.error(f"Generate memqa relation failed: {str(e)}")
            self.progress.mark_step_status(ProcessStep.GENERATE_MEMQA_RELATION, Status.FAILED)
            return False

    def generate_memqa_description(self) -> bool:
        """Generate memqa description"""
        try:
            # Mark step as in progress
            self.progress.mark_step_status(ProcessStep.GENERATE_MEMQA_DESCRIPTION, Status.IN_PROGRESS)
            logger.info("Starting memqa description generation...")

            logger.info("Generating memqa description...")
            description = DescriptionData()
            description.run()
            logger.info("Successfully generated memqa description")

            self.progress.mark_step_status(ProcessStep.GENERATE_MEMQA_DESCRIPTION, Status.COMPLETED)

            return True

        except Exception as e:
            logger.error(f"Generate memqa description failed: {str(e)}")
            self.progress.mark_step_status(ProcessStep.GENERATE_MEMQA_DESCRIPTION, Status.FAILED)
            return False

    def generate_memqa_diversity(self) -> bool:
        """Generate memqa diversity"""
        try:
            # Mark step as in progress
            self.progress.mark_step_status(ProcessStep.GENERATE_MEMQA_DIVERSITY, Status.IN_PROGRESS)
            logger.info("Starting memqa diversity generation...")

            logger.info("Generating memqa diversity...")
            diversity = DiversityData()
            diversity.run()
            logger.info("Successfully generated memqa diversity")

            self.progress.mark_step_status(ProcessStep.GENERATE_MEMQA_DIVERSITY, Status.COMPLETED)

            return True

        except Exception as e:
            logger.error(f"Generate memqa diversity failed: {str(e)}")
            self.progress.mark_step_status(ProcessStep.GENERATE_MEMQA_DIVERSITY, Status.FAILED)
            return False

    def synthetic_data_generation(self) -> bool:
        """Generate stage3 notes"""
        try:
            # Mark step as in progress
            self.progress.mark_step_status(ProcessStep.SYNTHETIC_DATA_GENERATION, Status.IN_PROGRESS)
            logger.info("Starting stage3 notes generation...")

            logger.info("Generating stage3 notes...")

            generator = SyntheticDataGenerator(
                language=self.language
            )
            os.makedirs("resources/data/stage3", exist_ok=True)
            synthetic_data_output_path = "resources/data/stage3/synthetic_data_with_notes_questions.json"
            generated_json = generator.generate_synthetic_data(output_file=synthetic_data_output_path)
            notes_answer_path = "resources/data/stage3/synthetic_data_with_notes_answers.json"
            success = generator.generate_notes_answers(generated_json, output_file=notes_answer_path)

            if success:
                logger.info("Synthetic data generation completed successfully!")
            else:
                logger.info("Synthetic data generation failed. Check logs for details.")

            self.progress.mark_step_status(ProcessStep.SYNTHETIC_DATA_GENERATION, Status.COMPLETED)

            return True

        except Exception as e:
            logger.error(f"Generate stage3 notes failed: {str(e)}")
            self.progress.mark_step_status(ProcessStep.SYNTHETIC_DATA_GENERATION, Status.FAILED)
            return False

    def synthetic_no_notes_data_generation(self) -> bool:
        """Generate stage3 no notes"""
        try:
            # Mark step as in progress
            self.progress.mark_step_status(ProcessStep.SYNTHETIC_NO_NOTES_DATA_GENERATION, Status.IN_PROGRESS)
            logger.info("Starting stage3 notes generation...")

            logger.info("Generating stage3 notes...")
            with open(
                    "resources/data/stage3/synthetic_data_with_notes_answers.json",
                    "r") as f:
                with_notes_data = json.load(f)

            len_of_no_note_data = len(with_notes_data) // 8

            generator = NoNoteDataGenerator(
                language=self.language
            )

            logger.info(f"开始生成 {len_of_no_note_data} 个无笔记问题...")
            questions = generator.gen_no_notes_questions(
                length=len_of_no_note_data,
                max_workers=10
            )
            logger.info(f"成功生成 {len(questions)} 个问题")

            logger.info("开始为问题生成回答...")
            answers = generator.gen_no_notes_answers(
                questions=questions,
                max_workers=10
            )
            logger.info(f"成功生成 {len(answers)} 个回答")
            logger.info("数据生成完成!")

            logger.info("Successfully generated stage3 notes")

            self.progress.mark_step_status(ProcessStep.SYNTHETIC_NO_NOTES_DATA_GENERATION, Status.COMPLETED)

            return True

        except Exception as e:
            logger.error(f"Generate stage3 notes failed: {str(e)}")
            self.progress.mark_step_status(ProcessStep.SYNTHETIC_NO_NOTES_DATA_GENERATION, Status.FAILED)
            return False

    def convert_data(self):
        try:
            # Mark step as in progress
            self.progress.mark_step_status(ProcessStep.CONVERT_DATA, Status.IN_PROGRESS)
            logger.info("Starting data conversion...")

            logger.info("Converting data...")
            convert_standard_data()
            logger.info("Successfully converted data")

            self.progress.mark_step_status(ProcessStep.CONVERT_DATA, Status.COMPLETED)

            return True

        except Exception as e:
            logger.error(f"Data conversion failed: {str(e)}")

    def data_filtering(self) -> bool:
        """Filter and assess quality of training data using Ollama Gemma"""
        try:
            # Mark step as in progress
            self.progress.mark_step_status(ProcessStep.DATA_FILTERING, Status.IN_PROGRESS)
            logger.info("Starting data filtering with Ollama Gemma...")

            # Import the MergedDataJudge
            from lpm_kernel.L2.merged_data_judge import MergedDataJudge
            
            # Get user biography for context
            from lpm_kernel.base.database_operate import get_latest_global_bio
            user_bio = get_latest_global_bio().content_third_view if get_latest_global_bio() else ""
            
            # Get filtering parameters from TrainingParamsManager
            training_params = TrainingParamsManager.get_latest_training_params()
            filtering_model = training_params.get('data_filtering_model', 'gemma:2b')
            max_workers = training_params.get('data_filtering_workers', 5)
            keep_ratio = training_params.get('data_filtering_keep_ratio', 0.8)
            
            # Log filtering parameters
            logger.info(f"Data filtering parameters:")
            logger.info(f"  - User bio length: {len(user_bio)} characters")
            logger.info(f"  - User bio preview: {user_bio[:200]}{'...' if len(user_bio) > 200 else ''}")
            logger.info(f"  - Filtering model: {filtering_model}")
            logger.info(f"  - Keep ratio: {keep_ratio * 100:.0f}% ({keep_ratio})")
            logger.info(f"  - Max workers: {max_workers}")
            
            # Initialize the judge with selected model
            judge = MergedDataJudge(
                model_name=filtering_model,
                ollama_host="http://localhost:11434",
                user_bio=user_bio
            )
            
            # Define input and output paths
            merged_json_path = "resources/data/merged.json"
            # filtered_output_path = "resources/data/filtered_merged.json"
            
            # Check if merged.json exists
            if not os.path.exists(merged_json_path):
                logger.error(f"Merged data file not found: {merged_json_path}")
                self.progress.mark_step_status(ProcessStep.DATA_FILTERING, Status.FAILED)
                return False
            
            # Perform data filtering
            logger.info("Starting data quality assessment and filtering...")
            judge.filter_and_score_data_concurrent(
                merged_json_path=merged_json_path,
                output_path=merged_json_path,
                user_bio=user_bio,
                keep_ratio=keep_ratio,
                max_workers=max_workers
            )
            
            # Replace the original merged.json with filtered data
            # import shutil
            # shutil.move(filtered_output_path, merged_json_path)
            logger.info(f"Data filtering completed. Filtered data saved to {merged_json_path}")
            
            # Release Ollama models from memory to free up VRAM for training
            logger.info("Releasing Ollama models from memory...")
            try:
                judge.cleanup()
                logger.info("✅ Successfully released Ollama models from memory")
            except Exception as e:
                logger.warning(f"⚠️ Could not release Ollama models: {str(e)}")
            
            self.progress.mark_step_status(ProcessStep.DATA_FILTERING, Status.COMPLETED)
            return True

        except Exception as e:
            logger.error(f"Data filtering failed: {str(e)}")
            self.progress.mark_step_status(ProcessStep.DATA_FILTERING, Status.FAILED)
            return False

    def train(self) -> bool:
        """Start model training"""
        try:
            # Mark step as in progress
            self.progress.mark_step_status(ProcessStep.TRAIN, Status.IN_PROGRESS)

            # Get paths for the model
            paths = self._get_model_paths(self.model_name)

            # Check if the model directory exists and has the necessary files
            config_file = os.path.join(paths["base_path"], "config.json")
            if not os.path.exists(paths["base_path"]) or not os.path.exists(config_file):
                logger.info(f"Model '{self.model_name}' needs to be downloaded or is missing config.json")
                # Call model_download to download the model
                download_success = self.model_download()
                if not download_success:
                    logger.error(f"Failed to download model '{self.model_name}'")
                    self.progress.mark_step_status(ProcessStep.MODEL_DOWNLOAD, Status.FAILED)
                    return False

            # Prepare log directory and file
            log_dir = os.path.join(os.getcwd(), "logs")
            os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(log_dir, "train", "train.log")
            logger.info(f"Log file path: {log_path}")

            # Ensure output directory exists
            os.makedirs(paths["personal_dir"], exist_ok=True)

            # Set USER_NAME environment variable
            os.environ["USER_NAME"] = LoadService.get_current_upload_name()
            logger.info(f"USER_NAME environment variable set: {os.environ['USER_NAME']}")

            script_path = os.path.join(os.getcwd(), "lpm_kernel/L2/train_for_user.sh")

            # First start monitoring progress in a separate thread
            logger.info("Starting monitoring thread first...")
            monitor_thread = threading.Thread(
                target=self._monitor_training_progress,
                args=(log_path,),
                daemon=True
            )
            monitor_thread.start()

            # Allow a moment for the monitoring thread to initialize
            time.sleep(1)

            # Then directly execute training process (blocking)
            logger.info("Now starting training process (blocking)...")
            training_result = self._start_training(script_path, log_path)

            if not training_result:
                logger.error("Training process failed to start")
                self.progress.mark_step_status(ProcessStep.TRAIN, Status.FAILED)
                return False

            # Wait for the monitoring thread to finish
            logger.info("Training process completed, waiting for monitoring to finish...")
            monitor_thread.join(timeout=10)  # Wait up to 10 seconds for monitor to finish

            # Check if the training was successful by checking the returncode
            if hasattr(self, 'training_result') and self.training_result:
                if self.training_result.get('returncode', 1) != 0:
                    error_msg = f"Training failed: {self.training_result.get('error', 'Unknown error')}"
                    logger.error(error_msg)
                    self.progress.mark_step_status(ProcessStep.TRAIN, Status.FAILED)
                    return False

            return True

        except Exception as e:
            logger.error(f"Failed to start training: {str(e)}")
            self.progress.mark_step_status(ProcessStep.TRAIN, Status.FAILED)
            return False

    def _get_model_paths(self, model_name):
        """Get all relevant paths for a model and set environment variables

        Args:
            model_name: Model name

        Returns:
            Dictionary containing all related paths:
            - base_path: Base model path
            - personal_dir: Personal trained model output directory
            - merged_dir: Merged model output directory
            - gguf_dir: GGUF model output directory
        """
        base_dir = os.getcwd()
        paths = {
            "base_path": os.path.join(base_dir, "resources/L2/base_models", model_name),
            "personal_dir": os.path.join(base_dir, "resources/model/output/personal_model", model_name),
            "merged_dir": os.path.join(base_dir, "resources/model/output/merged_model", model_name),
            "gguf_dir": os.path.join(base_dir, "resources/model/output/gguf", model_name)
        }

        # Ensure all directories exist
        for path in paths.values():
            os.makedirs(path, exist_ok=True)

        # Set environment variables
        os.environ["MODEL_BASE_PATH"] = paths["base_path"]
        os.environ["MODEL_PERSONAL_DIR"] = paths["personal_dir"]
        os.environ["MODEL_MERGED_DIR"] = paths["merged_dir"]
        os.environ["MODEL_GGUF_DIR"] = paths["gguf_dir"]

        # Log environment variables
        logger.info("Set environment variables:")
        logger.info(f"MODEL_BASE_PATH: {paths['base_path']}")
        logger.info(f"MODEL_PERSONAL_DIR: {paths['personal_dir']}")
        logger.info(f"MODEL_MERGED_DIR: {paths['merged_dir']}")
        logger.info(f"MODEL_GGUF_DIR: {paths['gguf_dir']}")

        return paths

    def _start_training(self, script_path, log_path):
        """Start training process

        Args:
            script_path: Path to training script
            log_path: Path to log file

        Returns:
            bool: True if the training process started successfully, False otherwise
        """
        try:
            # Reset stop flag before starting
            self.is_stopped = False

            # Get the latest training parameters from the class
            params_manager = TrainingParamsManager()
            training_params = params_manager.get_latest_training_params()
            learning_rate = training_params.get("learning_rate")
            num_train_epochs = training_params.get("number_of_epochs")
            concurrency_threads = training_params.get("concurrency_threads")
            data_synthesis_mode = training_params.get("data_synthesis_mode")
            use_cuda = training_params.get("use_cuda", False)
            is_cot = training_params.get("is_cot", False)

            # Log training parameters
            logger.info("Training parameters from latest settings:")
            logger.info(f"  Learning rate: {learning_rate}")
            logger.info(f"  Number of epochs: {num_train_epochs}")
            logger.info(f"  Concurrency threads: {concurrency_threads}")
            logger.info(f"  Data synthesis mode: {data_synthesis_mode}")
            logger.info(f"  Use CUDA: {use_cuda}")
            logger.info(f"  Is CoT: {is_cot}")

            # Prepare arguments for the script
            # Build command line arguments, need to include script path as the first parameter
            cmd = [
                script_path,
                "--lr", str(learning_rate),
                "--epochs", str(num_train_epochs),
                "--threads", str(concurrency_threads),
                "--mode", str(data_synthesis_mode),
                "--cuda", str(use_cuda),
                "--is_cot", str(is_cot)
            ]

            # Ensure log directory exists
            os.makedirs(os.path.dirname(log_path), exist_ok=True)

            # Set environment variables to improve tqdm output
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"  # Force Python to be unbuffered
            env["FORCE_COLOR"] = "1"  # Force colored output
            env["TQDM_FORCE_TTY"] = "1"  # Force tqdm to use TTY features

            # Ensure log directory exists
            log_dir = os.path.dirname(log_path)
            os.makedirs(log_dir, exist_ok=True)

            # Open log file
            log_file = open(log_path, "ab")

            # Use subprocess.Popen to directly execute the training script, redirecting output to file
            process = subprocess.Popen(
                cmd,
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                bufsize=0,  # Unbuffered
            )
            self.process = process
            self.current_pid = process.pid
            logger.info(f"Training process started with PID: {self.current_pid}")

            # Wait for process to finish directly (blocking)
            logger.info("Waiting for training process to complete...")
            return_code = process.wait()

            # Close log file
            log_file.close()

            # Save results for train method to check
            self.training_result = {
                "returncode": return_code,
                "error": f"Execution failed, return code: {return_code}" if return_code != 0 else None
            }

            if return_code != 0:
                logger.error(f"Command execution failed, return code: {return_code}")
                return False
            else:
                logger.info(f"Command execution successful, return code: {return_code}")

            return True

        except Exception as e:
            logger.error(f"Failed to start training process: {str(e)}")
            return False

    def _monitor_training_progress(self, log_file) -> bool:
        """Monitor training progress"""
        try:
            last_update_time = time.time()
            progress_file = os.path.join(os.path.dirname(TRAIN_LOG_FILE), "train_progress.json")
            while True:
                try:
                    percentage = 0.0
                    current_step = 0
                    total_steps = 100

                    if os.path.exists(progress_file):
                        try:
                            import json
                            with open(progress_file, 'r') as f:
                                progress_data = json.load(f)
                                percentage = progress_data.get("percentage", 0.0)
                                current_step = progress_data.get("current_step", 0)
                                total_steps = progress_data.get("total_steps", 100)
                        except Exception as e:
                            logger.error(f"Error reading progress file: {str(e)}")

                    current_time = time.time()
                    if current_time - last_update_time >= 1.0:
                        if percentage == 100.0:
                            self.progress.mark_step_status(ProcessStep.TRAIN, Status.COMPLETED)
                            return True

                        self._update_progress("training_to_create_second_me", "train", int(percentage / 3),
                                              f"Current step: {current_step}/{total_steps}")
                        last_update_time = current_time

                    time.sleep(1)

                except Exception as e:
                    logger.error(f"Error in progress monitoring: {str(e)}")
                    time.sleep(1)
                    continue

        except Exception as e:
            logger.error(f"Failed to monitor training progress: {str(e)}")
            self.progress.mark_step_status(ProcessStep.TRAIN, Status.FAILED)
            return False

    def _update_progress(self, stage: str, step: str, percentage: float, message: str, file_name: Optional[str] = None):
        """Update progress for any stage and step"""
        try:
            self.progress.progress.update_progress(
                stage,  # stage
                step,  # step
                Status.IN_PROGRESS,
                percentage,
                file_name  # Pass file name to update_progress method
            )
            logger.info(f"Progress updated: {percentage}% - {message}")
        except Exception as e:
            logger.error(f"Progress callback error: {str(e)}")

    def _monitor_model_download(self) -> bool:
        """Monitor model download progress"""
        try:
            # log_dir = os.path.join(os.getcwd(), "logs")
            # log_file = os.path.join(log_dir, "model_download.log")
            log_file = TRAIN_LOG_FILE

            # Initialize last_position to the end of file to only process new content
            try:
                with open(log_file, 'r') as f:
                    f.seek(0, 2)  # Move to the end of file
                    last_position = f.tell()
            except FileNotFoundError:
                # If file doesn't exist yet, start from beginning when it's created
                last_position = 0

            # Variables to track download status
            current_file = ""
            file_size = 0
            total_size = 0  # Total size of all files
            file_sizes = {}  # Dictionary to store file sizes
            last_update_time = time.time()

            while True:
                try:
                    # Read new log content
                    with open(log_file, 'r') as f:
                        f.seek(last_position)
                        new_lines = f.readlines()
                        last_position = f.tell()

                    for line in new_lines:
                        line = line.strip()

                        # Check for download start
                        if "Starting download of model:" in line:
                            logger.info("Model download started")
                            continue

                        # Get file size information when a download starts
                        if "Starting download of file:" in line:
                            match = re.search(r"Starting download of file: (.+) \(Size: ([\d\.]+) MB\)", line)
                            if match:
                                current_file = match.group(1)
                                file_size = float(match.group(2))
                                file_sizes[current_file] = file_size
                                total_size = sum(file_sizes.values())
                                # logger.info(f"Starting download of {current_file} ({file_size} MB)")

                        # Track file download progress
                        if "Downloaded" in line and "MB /" in line:
                            match = re.search(r"File (.+): Downloaded ([\d\.]+) MB / ([\d\.]+) MB \(([\d\.]+)%\)", line)
                            if match:
                                file_name = match.group(1)
                                downloaded_mb = float(match.group(2))
                                total_mb = float(match.group(3))
                                percentage = float(match.group(4))

                                # Update file size if it was updated (especially for model.safetensors)
                                if total_mb > file_sizes.get(file_name, 0):
                                    file_sizes[file_name] = total_mb
                                    total_size = sum(file_sizes.values())

                                # Calculate overall progress
                                if total_size > 0:
                                    # Sum up all downloaded data
                                    completed_files_size = sum(
                                        [file_sizes.get(f, 0) for f in file_sizes if f != file_name])
                                    current_file_downloaded = (percentage / 100.0) * total_mb
                                    overall_downloaded = completed_files_size + current_file_downloaded
                                    current_progress = (overall_downloaded / total_size) * 100
                                    current_progress = min(99.0, current_progress)  # Cap at 99% until fully complete
                                    # Update progress at most once per second
                                    current_time = time.time()
                                    if current_time - last_update_time >= 3.0:
                                        self._update_progress(
                                            "downloading_the_base_model",
                                            "model_download",
                                            current_progress,
                                            f"Overall: {current_progress:.1f}% - Downloading {file_name}: {percentage}% ({downloaded_mb:.1f}/{total_mb:.1f} MB)",
                                            file_name
                                        )
                                        last_update_time = current_time

                        if "Model downloaded successfully" in line:
                            self.progress.mark_step_status(ProcessStep.MODEL_DOWNLOAD, Status.COMPLETED)
                            logger.info("Model download completed")
                            return True

                    # Briefly pause to avoid excessive CPU usage
                    time.sleep(0.1)

                except IOError as e:
                    logger.error(f"Failed to read log file: {str(e)}")
                    time.sleep(0.1)
                    continue

        except Exception as e:
            logger.error(f"Failed to monitor model download progress: {str(e)}")
            return False

    def merge_weights(self) -> bool:
        """Merge weights"""
        try:
            # Mark step as in progress
            self.progress.mark_step_status(ProcessStep.MERGE_WEIGHTS, Status.IN_PROGRESS)

            paths = self._get_model_paths(self.model_name)

            # Check if model exists
            if not os.path.exists(paths["base_path"]):
                logger.error(f"Model '{self.model_name}' does not exist, please download first")
                self.progress.mark_step_status(ProcessStep.MERGE_WEIGHTS, Status.FAILED)
                return False

            # Check if training output exists
            if not os.path.exists(paths["personal_dir"]):
                return jsonify(APIResponse.error(
                    message=f"Model '{self.model_name}' training output does not exist, please train model first",
                    code=400
                ))

            # Ensure merged output directory exists
            os.makedirs(paths["merged_dir"], exist_ok=True)

            script_path = os.path.join(
                os.getcwd(), "lpm_kernel/L2/merge_weights_for_user.sh"
            )
            log_path = os.path.join(os.getcwd(), "logs", f"merge_weights_{self.model_name}.log")

            # Ensure log directory exists
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            # Use script executor to execute merge script
            script_executor = ScriptExecutor()
            result = script_executor.execute(
                script_path=script_path, script_type="merge_weights", log_file=log_path
            )

            logger.info(f"Weight merge task result: {result}")

            # Check if script execution was successful
            if result.get('returncode', 1) != 0:
                error_msg = f"Merge weights failed: {result.get('error', 'Unknown error')}"
                logger.error(error_msg)
                self.progress.mark_step_status(ProcessStep.MERGE_WEIGHTS, Status.FAILED)
                return False

            # Check if merged model files exist
            config_path = os.path.join(paths["merged_dir"], "config.json")
            if not os.path.exists(config_path):
                error_msg = f"Merged model files not found in {paths['merged_dir']}"
                logger.error(error_msg)
                self.progress.mark_step_status(ProcessStep.MERGE_WEIGHTS, Status.FAILED)
                return False

            logger.info("Weight merge completed successfully")
            self.progress.mark_step_status(ProcessStep.MERGE_WEIGHTS, Status.COMPLETED)
            return True

        except Exception as e:
            self.progress.mark_step_status(ProcessStep.MERGE_WEIGHTS, Status.FAILED)
            logger.error(f"Merge weights failed: {str(e)}")
            return False

    def convert_model(self) -> bool:
        """Convert model to GGUF format"""
        try:
            # Mark step as in progress
            self.progress.mark_step_status(ProcessStep.CONVERT_MODEL, Status.IN_PROGRESS)

            # Get paths for the model
            paths = self._get_model_paths(self.model_name)

            # Check if merged model exists
            merged_model_dir = paths["merged_dir"]
            logger.info(f"Merged model path: {merged_model_dir}")
            if not os.path.exists(merged_model_dir):
                logger.error(f"Model '{self.model_name}' merged output does not exist, please merge model first")
                self.progress.mark_step_status(ProcessStep.CONVERT_MODEL, Status.FAILED)
                return False

            # Get GGUF output directory
            gguf_dir = paths["gguf_dir"]
            logger.info(f"GGUF output directory: {gguf_dir}")

            # Generate timestamp for the filename
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            gguf_filename = f"{timestamp}.gguf"

            script_path = os.path.join(os.getcwd(), "lpm_kernel/L2/convert_hf_to_gguf.py")
            gguf_path = os.path.join(gguf_dir, gguf_filename)
            logger.info(f"GGUF output path: {gguf_path}")

            # Get training parameters from TrainingParamsManager
            from ..trainprocess.training_params_manager import TrainingParamsManager
            training_params = TrainingParamsManager.get_latest_training_params()
            logger.info(f"Retrieved training parameters: {training_params}")

            # Save training parameters to a JSON file in the GGUF directory
            training_params_path = os.path.join(gguf_dir, f"{timestamp}.json")
            try:
                # 添加模型路径到训练参数
                training_params["model_path"] = gguf_path

                with open(training_params_path, 'w', encoding='utf-8') as f:
                    json.dump(training_params, f, indent=2)
                logger.info(f"Training parameters saved to {training_params_path}")
            except Exception as e:
                logger.error(f"Failed to save training parameters: {str(e)}")
                self.progress.mark_step_status(ProcessStep.CONVERT_MODEL, Status.FAILED)
                return False

            # Build parameters
            args = [
                merged_model_dir,
                "--outfile",
                gguf_path,
                "--outtype",
                "f16",
            ]
            logger.info(f"Parameters: {args}")

            # Ensure GGUF output directory exists
            os.makedirs(os.path.dirname(gguf_path), exist_ok=True)

            # Use script executor to execute conversion script
            script_executor = ScriptExecutor()
            result = script_executor.execute(
                script_path=script_path,
                script_type="convert_model",
                args=args
            )

            # Model conversion completed
            try:
                with DatabaseSession.session() as session:
                    update_count = session.query(Memory).filter(Memory.status == "active").update(
                        {"is_trained": True},
                        synchronize_session=False  # 不同步会话状态，提高性能
                    )

                    # 提交更改
                    session.commit()
                logger.info(f"Updated training status for {update_count} memory records")
            except Exception as e:
                logger.error(f"Failed to update memory training status: {str(e)}", exc_info=True)
                self.progress.mark_step_status(ProcessStep.CONVERT_MODEL, Status.FAILED)
                return False

            logger.info(f"Model conversion result: {result}")

            # Check if script execution was successful
            if result.get('returncode', 1) != 0:
                error_msg = f"Model conversion failed: {result.get('error', 'Unknown error')}"
                logger.error(error_msg)
                self.progress.mark_step_status(ProcessStep.CONVERT_MODEL, Status.FAILED)
                return False

            # Check if GGUF model file exists
            if not os.path.exists(gguf_path):
                error_msg = f"GGUF model file not found at {gguf_path}"
                logger.error(error_msg)
                self.progress.mark_step_status(ProcessStep.CONVERT_MODEL, Status.FAILED)
                return False

            logger.info("Model conversion completed successfully")
            self.progress.mark_step_status(ProcessStep.CONVERT_MODEL, Status.COMPLETED)
            return True

        except Exception as e:
            self.progress.mark_step_status(ProcessStep.CONVERT_MODEL, Status.FAILED)
            logger.error(f"Convert model failed: {str(e)}")
            return False

    def check_training_condition(self) -> bool:
        """
        Check if the conditions for training are met
        Returns:
            bool: True if conditions are met, False otherwise
        """
        try:
            # Check if there are any documents that need embedding
            if document_service.check_all_documents_embeding_status():
                logger.warning("Cannot start training: There are documents that need embedding process first")
                return False
            return True
        except Exception as e:
            logger.error(f"Error checking training conditions: {str(e)}", exc_info=True)
            if self.progress.progress.current_stage:
                current_step = self.progress.progress.data["current_stage"]
                current_stage = next((s for s in self.progress.progress.data["stages"] if s["name"] == current_step),
                                     None)
                if current_stage and current_stage["current_step"]:
                    step = ProcessStep(current_stage["current_step"].lower().replace(" ", "_"))
                    self.progress.mark_step_status(step, Status.FAILED)
            return False

    def start_process(self) -> bool:
        """Start training process"""
        try:
            self.is_stopped = False
            self.current_pid = os.getpid()  # Store the PID
            logger.info(f"Training process started with PID: {self.current_pid}")
            ordered_steps = ProcessStep.get_ordered_steps()

            # Get the last successfully completed step
            last_successful_step = self.progress.get_last_successful_step()
            logger.info(
                f"Last successfully completed step: {last_successful_step.value if last_successful_step else 'None'}")
            start_index = 0
            if last_successful_step:
                start_index = ordered_steps.index(last_successful_step) + 1

            # Start executing from the step after the last successful one
            for step in ordered_steps[start_index:]:
                self.current_step = step
                if self.is_stopped:
                    logger.info("Training process aborted during step")
                    self.progress.mark_step_status(step, Status.SUSPENDED)
                    break  # If stop is requested, exit the loop

                logger.info(f"Starting step: {step.value}")

                # Execute the corresponding method
                method_name = step.get_method_name()
                if not hasattr(self, method_name):
                    logger.error(f"Method {method_name} not found")
                    self.progress.mark_step_status(step, Status.FAILED)
                    return False

                method = getattr(self, method_name)
                success = method()

                if not success:
                    logger.error(f"Step {step.value} failed")
                    logger.info(f'Marking step as failed: stage={step.value}, step={step.value}')
                    self.progress.mark_step_status(step, Status.FAILED)
                    return False
                logger.info(f"Step {step.value} completed successfully")
                # self.progress.mark_step_status(step, Status.COMPLETED)
            if self.is_stopped:
                logger.info("Training process was stopped during a step")
            else:
                logger.info("Training process completed...")

            return True
        except Exception as e:
            logger.error(f"Exception occurred: {str(e)}", exc_info=True)
            if self.current_step:
                self.progress.mark_step_status(self.current_step, Status.FAILED)
            return False

    def reset_progress(self):
        """Save current progress

        This method saves the current progress to the progress file.
        """
        try:
            self.progress.reset_progress()
            logger.info("Progress saved successfully")
        except Exception as e:
            logger.error(f"Failed to save progress: {str(e)}", exc_info=True)

    def get_step_output_content(self, step_name: str = None) -> Optional[Dict]:
        """Get content of output file for a specific training step

        Args:
            step_name: Name of the step to get content for. Required parameter.

        Returns:
            Optional[Dict]: Content of the output file for the specified step, or None if not found
        """
        try:
            # 特殊处理从数据库获取的内容
            if step_name == "generate_biography" or step_name == "generate_shades":
                logger.info(f"Querying database for {step_name}")
                return query_l1_version_data(1)

            # If step_name is not provided, return None
            if not step_name:
                logger.warning(f"Step name not provided")
                return None
                
            # 检查是否在output_files字典中
            if step_name in output_files:
                file_path = output_files[step_name]
                
                # 特殊处理"From database"标记
                if file_path == "From database":
                    logger.info(f"Querying database for {step_name}")
                    return query_l1_version_data(1)
                
                # 检查文件是否存在
                if not os.path.exists(file_path):
                    logger.warning(f"File path does not exist: {file_path}")
                    return None
                
                # 读取并返回文件内容
                logger.info(f"Reading file content from: {file_path}")
                return read_file_content(file_path)
            else:
                # 如果不在字典中，尝试在进度数据中找到路径
                for stage in self.progress.progress.data["stages"]:
                    for step in stage["steps"]:
                        step_key = step["name"].lower().replace(" ", "_")
                        if step_name == step_key and step.get("path"):
                            # 如果路径是"From database"，特殊处理
                            if step["path"] == "From database":
                                logger.info(f"Querying database for {step_name}")
                                return query_l1_version_data(1)
                                
                            # 构建完整路径
                            file_path = os.path.join(os.getcwd(), step["path"])
                            if not os.path.exists(file_path):
                                logger.warning(f"File path from progress data does not exist: {file_path}")
                                return None
                                
                            logger.info(f"Reading file content from progress data path: {file_path}")
                            return read_file_content(file_path)
                
                logger.warning(f"Step {step_name} not found in output_files or progress data")
                return None
        except Exception as e:
            logger.error(f"Error getting step output content: {str(e)}")
            return None

    def stop_process(self):
        """Stop training process

        Returns:
            bool: True if the process was stopped successfully, False otherwise
        """
        try:
            # Set the stop flag
            self.is_stopped = True
            logger.info("Training process has been requested to stop")
            # mark train stop
            if self.current_step == ProcessStep.TRAIN:
                self.progress.mark_step_status(ProcessStep.TRAIN, Status.SUSPENDED)

            # First check if we have the current process PID
            if not hasattr(self, 'current_pid') or not self.current_pid:
                logger.info("No active process PID found")
                if self.progress.progress.data["current_stage"]:
                    current_stage_name = self.progress.progress.data["current_stage"]
                    current_stage = next(
                        (s for s in self.progress.progress.data["stages"] if s["name"] == current_stage_name), None)
                    if current_stage and current_stage["current_step"]:
                        step = ProcessStep(current_stage["current_step"].lower().replace(" ", "_"))
                        self.progress.mark_step_status(step, Status.SUSPENDED)
                return True

            try:
                logger.info(f"Attempting to terminate process with PID: {self.current_pid}")

                # Check if the process exists
                if psutil.pid_exists(self.current_pid):
                    # Get the process object
                    process = psutil.Process(self.current_pid)

                    # Get all child processes
                    children = process.children(recursive=True)

                    # Terminate all child processes first
                    for child in children:
                        logger.info(f"Terminating child process with PID: {child.pid}")
                        try:
                            child.terminate()
                        except psutil.NoSuchProcess:
                            pass

                    # Wait for children to terminate
                    gone, still_alive = psutil.wait_procs(children, timeout=3)

                    # Kill any remaining children
                    for child in still_alive:
                        logger.info(f"Killing child process with PID: {child.pid}")
                        try:
                            child.kill()
                        except psutil.NoSuchProcess:
                            pass

                    logger.info(f"All child processes of {self.current_pid} have been terminated")
                    gc.collect()
                    return True
                else:
                    logger.warning(f"Process with PID {self.current_pid} no longer exists")
                    return True
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess) as e:
                logger.error(f"Failed to terminate process: {str(e)}", exc_info=True)

        except Exception as e:
            logger.error(f"Error stopping training process: {str(e)}", exc_info=True)
            return False

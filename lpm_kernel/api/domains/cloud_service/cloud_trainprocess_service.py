import enum
import json
import multiprocessing
import os
import signal
import time
from pathlib import Path

from lpm_kernel.api.domains.cloud_service.cloud_process_step import CloudProcessStep
from lpm_kernel.api.domains.cloud_service.cloud_progress_holder import CloudProgressHolder, CloudStatus
from lpm_kernel.api.domains.cloud_service.service import CloudService
from lpm_kernel.api.domains.trainprocess.process_step import ProcessStep
from lpm_kernel.api.domains.trainprocess.trainprocess_service import TrainProcessService
from lpm_kernel.common.repository.database_session import DatabaseSession
from lpm_kernel.configs.logging import get_train_process_logger
from lpm_kernel.models.memory import Memory

logger = get_train_process_logger()


class PrepareDataResult(enum.Enum):
    SUCCESS = "success"
    STOPPED = "stopped"
    ERROR = "error"


class CloudTrainProcessService(TrainProcessService):
    """Cloud training process service (singleton pattern)"""

    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, current_model_name: str, base_model, training_type, hyper_parameters):
        """Initialize cloud training process service"""
        # Initialize parent class
        super().__init__(current_model_name)

        # Override progress holder
        self.progress = CloudProgressHolder(current_model_name)

        # Initialize cloud-specific attributes
        self.training_data_path = None
        self.base_model = base_model
        self.training_type = training_type
        self.hyper_parameters = hyper_parameters
        self.model_name = current_model_name
        self.job_id = None
        
        # Load cloud training parameters for data filtering
        self.cloud_training_params = self._load_cloud_training_params()

        # For tracking data processing process
        self._data_processing_process = None
        self._data_processing_pid = None
        self._result_queue = None
        self._process_completed = None
        self._data_processing_result = None

        # For tracking task completion process
        self._wait_completion_process = None
        self._wait_completion_pid = None

        # Initialize cloud service
        self.cloud_service = CloudService()

    def _load_cloud_training_params(self):
        """Load cloud training parameters from file"""
        try:
            params_file = Path("data/cloud_progress/cloud_training_params.json")
            if params_file.exists():
                with open(params_file, "r", encoding="utf-8") as f:
                    params = json.load(f)
                logger.info(f"Loaded cloud training parameters: {params}")
                return params
            else:
                logger.warning("Cloud training parameters file not found, using default values")
                return {
                    "data_filtering_model": "gemma:2b",
                    "data_filtering_workers": 5,
                    "data_filtering_keep_ratio": 0.8
                }
        except Exception as e:
            logger.error(f"Failed to load cloud training parameters: {str(e)}")
            return {
                "data_filtering_model": "gemma:2b",
                "data_filtering_workers": 5,
                "data_filtering_keep_ratio": 0.8
            }

    @classmethod
    def get_instance(cls):
        """Get the current instance of CloudTrainProcessService
        
        Returns:
            CloudTrainProcessService: The singleton instance
        """

        if cls._instance is not None:
            return cls._instance

        try:

            params_file = Path("data/cloud_progress/cloud_training_params.json")
            if params_file.exists():
                with open(params_file, "r", encoding="utf-8") as f:
                    params = json.load(f)

                model_name = params.get("model_name")
                base_model = params.get("base_model")
                training_type = params.get("training_type", "efficient_sft")
                hyper_parameters = params.get("hyper_parameters", {})

                if model_name and base_model:
                    logger.info(f"Loaded training parameters for model {model_name} from file")

                    cls._instance = cls(current_model_name=model_name,
                                        base_model=base_model,
                                        training_type=training_type,
                                        hyper_parameters=hyper_parameters)
                    return cls._instance
                else:
                    logger.warning("Invalid training parameters in file: missing model_name or base_model")
        except Exception as e:
            logger.warning(f"Failed to load training parameters from file: {str(e)}")

        logger.warning("No valid training parameters found in file")
        return None

    def prepare_training_data(self) -> PrepareDataResult:
        """Prepare training data for cloud training"""
        try:
            logger.info("Starting training data preparation...")

            logger.info("Executing memory matrix activation steps...")
            stage_name = "activating_the_memory_matrix"
            stage = self.progress.progress.stage_map.get(stage_name)

            if self.progress.is_stage_completed(stage_name):
                logger.info(f"Stage '{stage_name}' already completed, skipping...")
            else:
                logger.info("Step 1.1: Listing documents...")
                if self.progress.is_step_completed(stage_name, "list_documents"):
                    logger.info("Step 'list_documents' already completed, skipping...")
                else:
                    if not super().list_documents():
                        logger.error("Failed to list documents")
                        self.progress.mark_step_status(ProcessStep.LIST_DOCUMENTS, CloudStatus.FAILED)
                        return PrepareDataResult.ERROR

                if stage:
                    stage["progress"] = 25.0
                    stage["status"] = CloudStatus.IN_PROGRESS
                    if len(stage["steps"]) > 0:
                        stage["steps"][0]["completed"] = True
                        stage["steps"][0]["status"] = CloudStatus.COMPLETED
                    logger.info(f"Updated {stage_name} progress to 25% after completing list_documents")
                    self._update_overall_progress()

                if self.is_stopped:
                    logger.info("Process has been stopped after completing list_documents, exiting.")
                    return PrepareDataResult.STOPPED

                logger.info("Step 1.2: Generating document embeddings...")
                if self.progress.is_step_completed(stage_name, "generate_document_embeddings"):
                    logger.info("Step 'generate_document_embeddings' already completed, skipping...")
                else:
                    # stage["current_step"] = ProcessStep.GENERATE_DOCUMENT_EMBEDDINGS
                    if not super().generate_document_embeddings():
                        logger.error("Failed to generate document embeddings")
                        self.progress.mark_step_status(ProcessStep.GENERATE_DOCUMENT_EMBEDDINGS, CloudStatus.FAILED)
                        return PrepareDataResult.ERROR

                if stage:
                    stage["progress"] = 50.0
                    if len(stage["steps"]) > 1:
                        stage["steps"][1]["completed"] = True
                        stage["steps"][1]["status"] = CloudStatus.COMPLETED
                    logger.info(f"Updated {stage_name} progress to 50% after completing generate_document_embeddings")
                    self._update_overall_progress()

                if self.is_stopped:
                    logger.info("Process has been stopped after completing generate_document_embeddings, exiting.")
                    return PrepareDataResult.STOPPED

                logger.info("Step 1.3: Processing chunks...")
                if self.progress.is_step_completed(stage_name, "process_chunks"):
                    logger.info("Step 'process_chunks' already completed, skipping...")
                else:
                    if not super().process_chunks():
                        logger.error("Failed to process chunks")
                        self.progress.mark_step_status(ProcessStep.CHUNK_DOCUMENT, CloudStatus.FAILED)
                        return PrepareDataResult.ERROR

                if stage:
                    stage["progress"] = 75.0
                    if len(stage["steps"]) > 2:
                        stage["steps"][2]["completed"] = True
                        stage["steps"][2]["status"] = CloudStatus.COMPLETED
                    logger.info(f"Updated {stage_name} progress to 75% after completing process_chunks")
                    self._update_overall_progress()

                if self.is_stopped:
                    logger.info("Process has been stopped after completing process_chunks, exiting.")
                    return PrepareDataResult.STOPPED

                logger.info("Step 1.4: Embedding chunks...")
                if self.progress.is_step_completed(stage_name, "chunk_embedding"):
                    logger.info("Step 'chunk_embedding' already completed, skipping...")
                else:
                    if not super().chunk_embedding():
                        logger.error("Failed to embed chunks")
                        self.progress.mark_step_status(ProcessStep.CHUNK_EMBEDDING, CloudStatus.FAILED)
                        return PrepareDataResult.ERROR

                # Update progress to 100% after completing first stage
                if stage:
                    stage["progress"] = 100.0
                    stage["status"] = CloudStatus.COMPLETED
                    # Update last step status
                    if len(stage["steps"]) > 3:
                        stage["steps"][3]["completed"] = True
                        stage["steps"][3]["status"] = CloudStatus.COMPLETED
                    logger.info(f"Updated {stage_name} progress to 100% and status to COMPLETED")
                    self._update_overall_progress()

                if self.is_stopped:
                    logger.info("Process has been stopped after completing chunk_embedding, exiting.")
                    return PrepareDataResult.STOPPED

            logger.info("Executing life narrative synthesis steps...")
            stage_name = "synthesize_your_life_narrative"
            stage = self.progress.progress.stage_map.get(stage_name)

            if self.progress.is_stage_completed(stage_name):
                logger.info(f"Stage '{stage_name}' already completed, skipping...")
            else:
                logger.info("Step 2.1: Extracting dimensional topics...")

                if self.progress.is_step_completed(stage_name, "extract_dimensional_topics"):
                    logger.info("Step 'extract_dimensional_topics' already completed, skipping...")
                else:
                    if not super().extract_dimensional_topics():
                        logger.error("Failed to extract dimensional topics")
                        self.progress.mark_step_status(ProcessStep.EXTRACT_DIMENSIONAL_TOPICS, CloudStatus.FAILED)
                        return PrepareDataResult.ERROR

                if stage:
                    stage["progress"] = 33.0
                    stage["status"] = CloudStatus.IN_PROGRESS

                    if len(stage["steps"]) > 0:
                        stage["steps"][0]["completed"] = True
                        stage["steps"][0]["status"] = CloudStatus.COMPLETED
                    logger.info(f"Updated {stage_name} progress to 33% after completing extract_dimensional_topics")
                    self._update_overall_progress()

                if self.is_stopped:
                    logger.info("Process has been stopped after completing extract_dimensional_topics, exiting.")
                    return PrepareDataResult.STOPPED

                logger.info("Step 2.2: Generating shades...")
                if self.progress.is_step_completed(stage_name, "generate_shades"):
                    logger.info("Step 'generate_shades' already completed, skipping...")
                else:
                    if not super().generate_shades():
                        logger.error("Failed to generate shades")
                        self.progress.mark_step_status(ProcessStep.GENERATE_SHADES, CloudStatus.FAILED)
                        return PrepareDataResult.ERROR

                if stage:
                    stage["progress"] = 66.0
                    if len(stage["steps"]) > 1:
                        stage["steps"][1]["completed"] = True
                        stage["steps"][1]["status"] = CloudStatus.COMPLETED
                    logger.info(f"Updated {stage_name} progress to 66% after completing generate_shades")
                    self._update_overall_progress()

                if self.is_stopped:
                    logger.info("Process has been stopped after completing generate_shades, exiting.")
                    return PrepareDataResult.STOPPED

                logger.info("Step 2.3: Generating biography...")
                if self.progress.is_step_completed(stage_name, "generate_biography"):
                    logger.info("Step 'generate_biography' already completed, skipping...")
                else:
                    if not super().generate_biography():
                        logger.error("Failed to generate biography")
                        self.progress.mark_step_status(ProcessStep.GENERATE_BIOGRAPHY, CloudStatus.FAILED)
                        return PrepareDataResult.ERROR

                if stage:
                    stage["progress"] = 100.0
                    stage["status"] = CloudStatus.COMPLETED
                    if len(stage["steps"]) > 2:
                        stage["steps"][2]["completed"] = True
                        stage["steps"][2]["status"] = CloudStatus.COMPLETED
                    logger.info(f"Updated {stage_name} progress to 100% after completing generate_biography")
                    self._update_overall_progress()

                if self.is_stopped:
                    logger.info("Process has been stopped after completing generate_biography, exiting.")
                    return PrepareDataResult.STOPPED

            logger.info("Executing memory reconstruction steps...")
            stage_name = "memory_reconstruction"
            stage = self.progress.progress.stage_map.get(stage_name)

            if self.progress.is_stage_completed(stage_name):
                logger.info(f"Stage '{stage_name}' already completed, skipping...")
            else:
                logger.info("Step 3.1: Generating base data...")
                if self.progress.is_step_completed(stage_name, "generate_base"):
                    logger.info("Step 'generate_base' already completed, skipping...")
                else:
                    # stage["current_step"] = ProcessStep.GENERATE_BASE
                    if not super().generate_base():
                        logger.error("Failed to generate base data")
                        self.progress.mark_step_status(ProcessStep.GENERATE_BASE, CloudStatus.FAILED)
                        return PrepareDataResult.ERROR

                if stage:
                    stage["progress"] = 100.0
                    stage["status"] = CloudStatus.COMPLETED
                    if len(stage["steps"]) > 0:
                        stage["steps"][0]["completed"] = True
                        stage["steps"][0]["status"] = CloudStatus.COMPLETED
                    logger.info(f"Updated {stage_name} progress to 100% after completing generate_base")
                    self._update_overall_progress()

                if self.is_stopped:
                    logger.info("Process has been stopped after completing generate_base, exiting.")
                    return PrepareDataResult.STOPPED

            logger.info("Executing deep comprehension steps...")
            stage_name = "deep_comprehension"
            stage = self.progress.progress.stage_map.get(stage_name)

            if self.progress.is_stage_completed(stage_name):
                logger.info(f"Stage '{stage_name}' already completed, skipping...")
            else:
                logger.info("Step 4.1: Generating bio QA...")
                if self.progress.is_step_completed(stage_name, "bio_qa_generation"):
                    logger.info("Step 'bio_qa_generation' already completed, skipping...")
                else:
                    # stage["current_step"] = ProcessStep.BIO_QA_GENERATION
                    if not super().bio_qa_generation():
                        logger.error("Failed to generate bio QA")
                        self.progress.mark_step_status(ProcessStep.BIO_QA_GENERATION, CloudStatus.FAILED)
                        return PrepareDataResult.ERROR

                if stage:
                    stage["progress"] = 16.7
                    stage["status"] = CloudStatus.IN_PROGRESS
                    if len(stage["steps"]) > 0:
                        stage["steps"][0]["completed"] = True
                        stage["steps"][0]["status"] = CloudStatus.COMPLETED
                    logger.info(f"Updated {stage_name} progress to 16.7% after completing bio_qa_generation")
                    self._update_overall_progress()

                if self.is_stopped:
                    logger.info("Process has been stopped after completing bio_qa_generation, exiting.")
                    return PrepareDataResult.STOPPED

                logger.info("Step 4.2: Generating wiki data...")
                if self.progress.is_step_completed(stage_name, "wiki_data_generation"):
                    logger.info("Step 'wiki_data_generation' already completed, skipping...")
                else:
                    try:
                        # stage["current_step"] = ProcessStep.WIKI_DATA_GENERATION
                        if not super().wiki_data_generation():
                            logger.error("Failed to generate wiki data")
                            self.progress.mark_step_status(ProcessStep.WIKI_DATA_GENERATION, CloudStatus.FAILED)
                            return PrepareDataResult.ERROR
                    except Exception as e:
                        logger.error(f"Wiki data generation failed: {str(e)}")
                        self.progress.mark_step_status(ProcessStep.WIKI_DATA_GENERATION, CloudStatus.FAILED)
                        return PrepareDataResult.ERROR

                if stage:
                    stage["progress"] = 33.4
                    if len(stage["steps"]) > 1:
                        stage["steps"][1]["completed"] = True
                        stage["steps"][1]["status"] = CloudStatus.COMPLETED
                    logger.info(f"Updated {stage_name} progress to 33.4% after completing wiki_data_generation")
                    self._update_overall_progress()

                if self.is_stopped:
                    logger.info("Process has been stopped after completing wiki_data_generation, exiting.")
                    return PrepareDataResult.STOPPED

                logger.info("Step 4.3: Generating MemQA entity...")
                if self.progress.is_step_completed(stage_name, "generate_memqa_entity"):
                    logger.info("Step 'generate_memqa_entity' already completed, skipping...")
                else:
                    # stage["current_step"] = ProcessStep.GENERATE_MEMQA_ENTITY
                    if not super().generate_memqa_entity():
                        logger.error("Failed to generate MemQA entity")
                        self.progress.mark_step_status(ProcessStep.GENERATE_MEMQA_ENTITY, CloudStatus.FAILED)
                        return PrepareDataResult.ERROR

                if stage:
                    stage["progress"] = 50.1
                    if len(stage["steps"]) > 2:
                        stage["steps"][2]["completed"] = True
                        stage["steps"][2]["status"] = CloudStatus.COMPLETED
                    logger.info(f"Updated {stage_name} progress to 50.1% after completing generate_memqa_entity")
                    self._update_overall_progress()

                if self.is_stopped:
                    logger.info("Process has been stopped after completing generate_memqa_entity, exiting.")
                    return PrepareDataResult.STOPPED

                logger.info("Step 4.4: Generating MemQA relation...")
                if self.progress.is_step_completed(stage_name, "generate_memqa_relation"):
                    logger.info("Step 'generate_memqa_relation' already completed, skipping...")
                else:
                    if not super().generate_memqa_relation():
                        logger.error("Failed to generate MemQA relation")
                        self.progress.mark_step_status(ProcessStep.GENERATE_MEMQA_RELATION, CloudStatus.FAILED)
                        return PrepareDataResult.ERROR

                if stage:
                    stage["progress"] = 66.8
                    if len(stage["steps"]) > 3:
                        stage["steps"][3]["completed"] = True
                        stage["steps"][3]["status"] = CloudStatus.COMPLETED
                    logger.info(f"Updated {stage_name} progress to 66.8% after completing generate_memqa_relation")
                    self._update_overall_progress()

                if self.is_stopped:
                    logger.info("Process has been stopped after completing generate_memqa_relation, exiting.")
                    return PrepareDataResult.STOPPED

                logger.info("Step 4.5: Generating MemQA description...")
                if self.progress.is_step_completed(stage_name, "generate_memqa_description"):
                    logger.info("Step 'generate_memqa_description' already completed, skipping...")
                else:
                    if not super().generate_memqa_description():
                        logger.error("Failed to generate MemQA description")
                        self.progress.mark_step_status(ProcessStep.GENERATE_MEMQA_DESCRIPTION, CloudStatus.FAILED)
                        return PrepareDataResult.ERROR

                if stage:
                    stage["progress"] = 83.5
                    if len(stage["steps"]) > 4:
                        stage["steps"][4]["completed"] = True
                        stage["steps"][4]["status"] = CloudStatus.COMPLETED
                    logger.info(f"Updated {stage_name} progress to 83.5% after completing generate_memqa_description")
                    self._update_overall_progress()

                if self.is_stopped:
                    logger.info("Process has been stopped after completing generate_memqa_description, exiting.")
                    return PrepareDataResult.STOPPED

                logger.info("Step 4.6: Generating MemQA diversity...")
                if self.progress.is_step_completed(stage_name, "generate_memqa_diversity"):
                    logger.info("Step 'generate_memqa_diversity' already completed, skipping...")
                else:
                    if not super().generate_memqa_diversity():
                        logger.error("Failed to generate MemQA diversity")
                        self.progress.mark_step_status(ProcessStep.GENERATE_MEMQA_DIVERSITY, CloudStatus.FAILED)
                        return PrepareDataResult.ERROR

                if stage:
                    stage["progress"] = 100.0
                    stage["status"] = CloudStatus.COMPLETED
                    if len(stage["steps"]) > 5:
                        stage["steps"][5]["completed"] = True
                        stage["steps"][5]["status"] = CloudStatus.COMPLETED
                    logger.info(f"Updated {stage_name} progress to 100% after completing generate_memqa_diversity")
                    self._update_overall_progress()

                if self.is_stopped:
                    logger.info("Process has been stopped after completing generate_memqa_diversity, exiting.")
                    return PrepareDataResult.STOPPED

            logger.info("Executing memory expansion steps...")
            stage_name = "memory_expansion"
            stage = self.progress.progress.stage_map.get(stage_name)

            if self.progress.is_stage_completed(stage_name):
                logger.info(f"Stage '{stage_name}' already completed, skipping...")
            else:
                logger.info("Step 5.1: Generating synthetic data...")
                if self.progress.is_step_completed(stage_name, "synthetic_data_generation"):
                    logger.info("Step 'synthetic_data_generation' already completed, skipping...")
                else:
                    if not super().synthetic_data_generation():
                        logger.error("Failed to generate synthetic data")
                        self.progress.mark_step_status(ProcessStep.SYNTHETIC_DATA_GENERATION, CloudStatus.FAILED)
                        return PrepareDataResult.ERROR

                if stage:
                    stage["progress"] = 33.3
                    stage["status"] = CloudStatus.IN_PROGRESS
                    if len(stage["steps"]) > 0:
                        stage["steps"][0]["completed"] = True
                        stage["steps"][0]["status"] = CloudStatus.COMPLETED
                    logger.info(f"Updated {stage_name} progress to 33.3% after completing synthetic_data_generation")
                    self._update_overall_progress()

                if self.is_stopped:
                    logger.info("Process has been stopped after completing synthetic_data_generation, exiting.")
                    return PrepareDataResult.STOPPED

                logger.info("Step 5.2: Generating synthetic no notes data...")
                if self.progress.is_step_completed(stage_name, "synthetic_no_notes_data_generation"):
                    logger.info("Step 'synthetic_no_notes_data_generation' already completed, skipping...")
                else:
                    if not super().synthetic_no_notes_data_generation():
                        logger.error("Failed to generate synthetic no notes data")
                        self.progress.mark_step_status(ProcessStep.SYNTHETIC_NO_NOTES_DATA_GENERATION,
                                                       CloudStatus.FAILED)
                        return PrepareDataResult.ERROR

                if stage:
                    stage["progress"] = 66.6
                    if len(stage["steps"]) > 1:
                        stage["steps"][1]["completed"] = True
                        stage["steps"][1]["status"] = CloudStatus.COMPLETED
                    logger.info(
                        f"Updated {stage_name} progress to 66.6% after completing synthetic_no_notes_data_generation")
                    self._update_overall_progress()

                if self.is_stopped:
                    logger.info(
                        "Process has been stopped after completing synthetic_no_notes_data_generation, exiting.")
                    return PrepareDataResult.STOPPED

                logger.info("Step 5.3: Converting data...")
                if self.progress.is_step_completed(stage_name, "convert_data"):
                    logger.info("Step 'convert_data' already completed, skipping...")
                else:
                    if not super().convert_data():
                        logger.error("Failed to convert data")
                        self.progress.mark_step_status(ProcessStep.CONVERT_DATA, CloudStatus.FAILED)
                        return PrepareDataResult.ERROR

                if stage:
                    stage["progress"] = 75.0
                    if len(stage["steps"]) > 2:
                        stage["steps"][2]["completed"] = True
                        stage["steps"][2]["status"] = CloudStatus.COMPLETED
                    logger.info(f"Updated {stage_name} progress to 75% after completing convert_data")
                    self._update_overall_progress()

                if self.is_stopped:
                    logger.info("Process has been stopped after completing convert_data, exiting.")
                    return PrepareDataResult.STOPPED

                logger.info("Step 5.4: Data filtering...")
                if self.progress.is_step_completed(stage_name, "data_filtering"):
                    logger.info("Step 'data_filtering' already completed, skipping...")
                else:
                    if not super().data_filtering():
                        logger.error("Failed to perform data filtering")
                        self.progress.mark_step_status(ProcessStep.DATA_FILTERING, CloudStatus.FAILED)
                        return PrepareDataResult.ERROR

                if stage:
                    stage["progress"] = 100.0
                    stage["status"] = CloudStatus.COMPLETED
                    if len(stage["steps"]) > 3:
                        stage["steps"][3]["completed"] = True
                        stage["steps"][3]["status"] = CloudStatus.COMPLETED
                    logger.info(f"Updated {stage_name} progress to 100% after completing data_filtering")
                    self._update_overall_progress()

                if self.is_stopped:
                    logger.info("Process has been stopped after completing data_filtering, exiting.")
                    return PrepareDataResult.STOPPED

            self._update_overall_progress()

            if self.is_stopped:
                logger.info("Data preparation completed current step, stopping as requested")
                return PrepareDataResult.STOPPED

            logger.info("Successfully generated all necessary data using parent class methods")
            return PrepareDataResult.SUCCESS
        except Exception as e:
            logger.error(f"Prepare training data failed: {str(e)}")
            current_stage = self.progress.get_progress().get("current_stage")
            if current_stage:
                for step in self.progress.get_progress().get("stages", []):
                    if step["name"].lower().replace(" ", "_") == current_stage and step["current_step"]:
                        stage_name = current_stage
                        self.progress.mark_step_status(stage_name, CloudStatus.FAILED)
                        break
            return PrepareDataResult.ERROR

    def data_filtering(self) -> bool:
        """Override data filtering to use cloud training parameters"""
        try:
            # Mark step as in progress
            self.progress.mark_step_status(ProcessStep.DATA_FILTERING, CloudStatus.IN_PROGRESS)
            logger.info("Starting data filtering with cloud training parameters...")

            # Import the MergedDataJudge
            from lpm_kernel.L2.merged_data_judge import MergedDataJudge
            
            # Get user biography for context
            from lpm_kernel.base.database_operate import get_latest_global_bio
            user_bio = get_latest_global_bio().content_third_view if get_latest_global_bio() else ""
            
            # Get filtering parameters from cloud training parameters
            filtering_model = self.cloud_training_params.get('data_filtering_model', 'gemma:2b')
            max_workers = self.cloud_training_params.get('data_filtering_workers', 5)
            keep_ratio = self.cloud_training_params.get('data_filtering_keep_ratio', 0.8)
            
            # Log filtering parameters
            logger.info(f"Cloud data filtering parameters:")
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
            
            # Check if merged.json exists
            if not os.path.exists(merged_json_path):
                logger.error(f"Merged data file not found: {merged_json_path}")
                self.progress.mark_step_status(ProcessStep.DATA_FILTERING, CloudStatus.FAILED)
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
            
            logger.info(f"Data filtering completed. Filtered data saved to {merged_json_path}")
            
            # Release Ollama models from memory to free up VRAM for training
            logger.info("Releasing Ollama models from memory...")
            try:
                judge.cleanup()
                logger.info("✅ Successfully released Ollama models from memory")
            except Exception as e:
                logger.warning(f"⚠️ Could not release Ollama models: {str(e)}")
            
            self.progress.mark_step_status(ProcessStep.DATA_FILTERING, CloudStatus.COMPLETED)
            return True

        except Exception as e:
            logger.error(f"Cloud data filtering failed: {str(e)}")
            self.progress.mark_step_status(ProcessStep.DATA_FILTERING, CloudStatus.FAILED)
            return False

    def start_process(self) -> bool:
        """Start the cloud training process using CloudService"""
        self.is_stopped = False
        self._data_processing_result = None

        self.current_pid = os.getpid()
        logger.info(f"Cloud training process started with PID: {self.current_pid}")
        logger.info(f"Using base_model: {self.base_model}, training_type: {self.training_type}")
        logger.info(f"CloudService initialized with API key: {self.cloud_service.api_key is not None}")

        logger.info("Step 1: Preparing training data...")

        try:
            if self.is_stopped:
                logger.info("Process has been stopped, will complete current stage and then stop")

            result = self.prepare_training_data()
            self._data_processing_result = result

            success = self._data_processing_result
            logger.info(f"Training data preparation result: {success}")

            if success == PrepareDataResult.SUCCESS:
                logger.info("Training data preparation completed successfully")
            elif success == PrepareDataResult.STOPPED:
                logger.info("Training data preparation stopped by user")
                return False
            elif success == PrepareDataResult.ERROR:
                logger.error("Failed to prepare training data")
                return False

            if self.is_stopped:
                logger.info("Process has been stopped after data preparation")
                return False

            deploy_success = self.cloud_deploy()
            logger.info(f"Cloud deploy result: {deploy_success}")
            if not deploy_success:
                logger.error("Failed to cloud deploy")
                return False

            return True
        except Exception as e:
            logger.error(f"Error in cloud training process: {str(e)}", exc_info=True)
            return False

    def cloud_deploy(self) -> bool:
        try:
            logger.info("Step 7: Uploading training data...")
            if self.is_stopped:
                logger.info("Process has been stopped, cancelling cloud deployment")
                return False

            self.progress.mark_step_status(CloudProcessStep.UPLOAD_TRAINING_DATA, CloudStatus.IN_PROGRESS)
            try:
                file_id = self.cloud_service.upload_training_file()
                logger.info(f"File upload result: file_id={file_id}")
            except Exception as e:
                logger.error(f"Exception during file upload: {str(e)}", exc_info=True)
                self.progress.mark_step_status(CloudProcessStep.UPLOAD_TRAINING_DATA, CloudStatus.FAILED)
                return False

            if not file_id:
                logger.error("Failed to upload training data")
                self.progress.mark_step_status(CloudProcessStep.UPLOAD_TRAINING_DATA, CloudStatus.FAILED)
                return False
            self.progress.mark_step_status(CloudProcessStep.UPLOAD_TRAINING_DATA, CloudStatus.COMPLETED)

            logger.info("Step 8: Creating fine-tune job...")
            self.progress.mark_step_status(CloudProcessStep.CREATE_FINE_TUNE_JOB, CloudStatus.IN_PROGRESS)

            try:
                success_id = self.cloud_service.create_fine_tune_job(
                    base_model=self.base_model,
                    training_type=self.training_type,
                    hyper_parameters=self.hyper_parameters
                )
                try:
                    params_dir = Path("data/cloud_progress")
                    params_dir.mkdir(parents=True, exist_ok=True)
                    job_file_path = params_dir / "job_id.json"

                    job_info = {
                        "job_id": success_id,
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "status": "completed"
                    }

                    with open(job_file_path, "w") as f:
                        json.dump(job_info, f, indent=2)

                    logger.info(f"Job ID information saved to {job_file_path}")
                except Exception as e:
                    logger.error(f"Failed to write job ID to file: {str(e)}", exc_info=True)

                logger.info(f"Create fine-tune job result: {success_id}")
            except Exception as e:
                logger.error(f"Exception during fine-tune job creation: {str(e)}", exc_info=True)
                self.progress.mark_step_status(CloudProcessStep.CREATE_FINE_TUNE_JOB, CloudStatus.FAILED)
                return False

            if success_id is None:
                logger.error("Failed to create fine-tune job")
                self.progress.mark_step_status(CloudProcessStep.CREATE_FINE_TUNE_JOB, CloudStatus.FAILED)
                return False

            self.job_id = success_id
            logger.info(f"Job ID set: {self.job_id}")

            self.progress.job_id = self.job_id
            self.progress.progress.data["job_id"] = self.job_id
            self.progress.mark_step_status(CloudProcessStep.CREATE_FINE_TUNE_JOB, CloudStatus.COMPLETED)

            logger.info("Step 9: Waiting for fine-tune job to complete...")

            self.progress.mark_step_status(CloudProcessStep.WAIT_FOR_FINE_TUNE_COMPLETION, CloudStatus.IN_PROGRESS)

            logger.info(f"Fine-tune job {self.job_id} has been created and is now running")

            # Start a separate process to monitor the job completion
            logger.info("Starting a separate process to monitor the job completion")

            current_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
            os.environ["BASE_DIR"] = current_dir

            self._wait_completion_process = multiprocessing.Process(
                target=self._wait_for_completion_process,
                args=(self.cloud_service, self.job_id)
            )
            self._wait_completion_process.daemon = True
            self._wait_completion_process.start()
            self._wait_completion_pid = self._wait_completion_process.pid
            logger.info(f"Job monitoring process started with PID: {self._wait_completion_pid}")

            self.progress.mark_step_status(CloudProcessStep.WAIT_FOR_FINE_TUNE_COMPLETION, CloudStatus.IN_PROGRESS)

            logger.info("Cloud training process completed successfully")
            return True
        except Exception as e:
            logger.error(f"Cloud training process failed: {str(e)}", exc_info=True)
            if self.current_step:
                self.progress.mark_step_status(self.current_step, CloudStatus.FAILED)
            return False

    def _wait_for_completion_process(self, cloud_service, job_id):
        try:
            def handle_sigterm(signum, frame):
                logger.info(f"Wait completion process received SIGTERM signal, exiting...")
                import sys
                sys.exit(0)

            signal.signal(signal.SIGTERM, handle_sigterm)

            logger.info(f"Async process: waiting for job {job_id} to complete")

            def progress_callback(status, progress, message):
                try:
                    logger.info(f"Progress update: {status}, {progress}%, {message}")

                    status_mapping = {
                        "IN_PROGRESS": CloudStatus.IN_PROGRESS,
                        "COMPLETED": CloudStatus.COMPLETED,
                        "FAILED": CloudStatus.FAILED,
                        "CANCELED": CloudStatus.CANCELED
                    }

                    cloud_status = status_mapping.get(status, CloudStatus.IN_PROGRESS)

                    self.progress.update_step_progress(
                        CloudProcessStep.WAIT_FOR_FINE_TUNE_COMPLETION,
                        progress,
                        message
                    )

                    if status in ["COMPLETED", "FAILED", "CANCELED"]:
                        self.progress.mark_step_status(
                            CloudProcessStep.WAIT_FOR_FINE_TUNE_COMPLETION,
                            cloud_status
                        )
                except Exception as e:
                    logger.error(f"Error in progress callback: {str(e)}", exc_info=True)

            success = cloud_service.wait_for_job_completion(
                job_id=job_id,
                progress_callback=progress_callback
            )

            if success:
                self.progress.update_message("Fine-tuning job completed successfully!")
                # Update is_trained flag for memory records after successful cloud training
                self.update_memory_training_status()
            else:
                logger.error(f"Fine-tuning job failed")
        except Exception as e:
            logger.error(f"Error in async wait thread: {str(e)}", exc_info=True)
            self.progress.mark_step_status(CloudProcessStep.WAIT_FOR_FINE_TUNE_COMPLETION, CloudStatus.FAILED)

    def update_memory_training_status(self):
        """Update is_trained flag for memory records after successful cloud training"""
        try:

            with DatabaseSession.session() as session:
                update_count = session.query(Memory).filter(Memory.status == "active").update(
                    {"is_trained": True},
                    synchronize_session=False
                )

                session.commit()
            logger.info(f"Updated training status for {update_count} memory records after cloud training")
        except Exception as e:
            logger.error(f"Failed to update memory training status: {str(e)}", exc_info=True)

    def _update_overall_progress(self):
        """Calculate and update the overall progress based on the stages' progress"""
        try:
            stages = self.progress.progress.data["stages"]
            total_stages = len(stages)
            completed_stages = 0
            total_progress = 0.0

            for stage in stages:
                total_progress += stage["progress"]
                if stage["status"] == CloudStatus.COMPLETED:
                    completed_stages += 1

            if total_stages > 0:
                overall_progress = total_progress / total_stages
            else:
                overall_progress = 0.0

            self.progress.progress.data["overall_progress"] = overall_progress
            logger.info(f"Updated overall progress to {overall_progress:.2f}%")

            if completed_stages == total_stages:
                self.progress.progress.data["status"] = CloudStatus.COMPLETED
                logger.info("All stages completed, setting overall status to COMPLETED")

            self.progress.save_progress()
        except Exception as e:
            logger.error(f"Error updating overall progress: {str(e)}")

    def stop_process(self) -> str:
        """Stop the cloud training process
        
        This method will attempt to stop the fine-tuning job if it's in progress,
        by deleting the job, and update the progress status accordingly.
        It will also wait for the current data processing step to complete before returning.
        
        Returns:
            str: A message indicating the status of the stop operation
        """
        try:
            logger.info(f"Attempting to stop cloud training process for model: {self.model_name}")

            if self.is_stopped:
                logger.info("Training process is already stopped, returning success")
                # 确保状态被设置为暂停
                if hasattr(self, 'progress') and self.progress and hasattr(self.progress, 'progress') and self.progress.progress and hasattr(self.progress.progress, 'data'):
                    self.progress.progress.data["status"] = CloudStatus.SUSPENDED
                    self.progress.save_progress()
                return 'success'
                
            self.is_stopped = True

            current_stage = self.progress.get_progress().get("current_stage")
            logger.info(f"Current stage when stopping: {current_stage}")
            current_step = None

            # Check if we're in the data synthesis stage
            is_data_synthesis_stage = False

            if current_stage:
                for stage in self.progress.get_progress().get("stages", []):
                    if stage["name"] == current_stage:
                        current_step_name = stage.get("current_step")
                        if current_step_name:
                            # 将current_step_name转换为小写并将空格替换为下划线
                            normalized_step_name = current_step_name.lower().replace(" ", "_")
                            logger.info(f"Normalized step name: {normalized_step_name}")
                            for step in ProcessStep:
                                if step.value == normalized_step_name:
                                    current_step = step
                                    logger.info(f"Found step in ProcessStep: {current_step}")
                                    is_data_synthesis_stage = True
                                    break

                        break

            logger.info(f"Current step when stopping: {current_step}")
            logger.info(f"Is data synthesis stage: {is_data_synthesis_stage}")

            # If we're in the data synthesis stage, check the step status
            if is_data_synthesis_stage and current_step:
                step_status = None
                current_stage_data = None

                for stage in self.progress.progress.data["stages"]:
                    if stage["name"] == current_stage:
                        current_stage_data = stage
                        logger.info(f"Found current stage data: {stage['name']}")
                        break

                if current_stage_data:
                    step_name = current_step.value if hasattr(current_step, 'value') else str(current_step)
                    for step in current_stage_data["steps"]:
                        normalized_step_name = step["name"].lower().replace(" ", "_")
                        if normalized_step_name == step_name:
                            step_status = step["status"]
                            logger.info(f"Found step status: {step_status}")
                            break

                    if step_status in [CloudStatus.COMPLETED, CloudStatus.FAILED]:
                        logger.info(f"Step {current_step.value} has status {step_status}, continuing with stop process")
                        self.progress.progress.data["status"] = CloudStatus.SUSPENDED
                        self.progress.save_progress()
                    else:
                        logger.info(f"Step {current_step.value} is still running, returning pending status")
                        self.progress.progress.data["status"] = CloudStatus.PENDING
                        self.progress.save_progress()
                        return "pending"

            if not self.job_id:
                try:
                    params_dir = Path("data/cloud_progress")
                    job_file_path = params_dir / "job_id.json"

                    if job_file_path.exists():
                        with open(job_file_path, "r") as f:
                            job_info = json.load(f)
                            if "job_id" in job_info:
                                self.job_id = job_info["job_id"]
                                logger.info(f"Retrieved job_id from file: {self.job_id}")
                except Exception as e:
                    logger.error(f"Failed to read job ID from file: {str(e)}", exc_info=True)

            if self.job_id:
                logger.info(f"Attempting to cancel fine-tune job: {self.job_id}")
                success = self.cloud_service.cancel_fine_tune_job(self.job_id)

                if success:
                    logger.info(f"Successfully canceled fine-tune job: {self.job_id}")
                else:
                    logger.error(f"Failed to cancel fine-tune job: {self.job_id}")
            else:
                logger.warning("No active fine-tune job found to delete")

            if self._wait_completion_process and self._wait_completion_process.is_alive():
                logger.info(f"Terminating wait completion process (PID: {self._wait_completion_pid})")
                try:
                    os.kill(self._wait_completion_pid, signal.SIGTERM)
                    self._wait_completion_process.join(timeout=5)
                    if self._wait_completion_process.is_alive():
                        logger.warning(f"Wait completion process did not terminate gracefully, forcing termination")
                        self._wait_completion_process.terminate()
                    logger.info(f"Wait completion process terminated successfully")
                except Exception as e:
                    logger.error(f"Error terminating wait completion process: {str(e)}", exc_info=True)

            # If not in data synthesis stage, set the cloud process steps to pending
            if not is_data_synthesis_stage:
                logger.info("Not in data synthesis stage, setting cloud process steps to pending status")

                # Set the three specific cloud process steps to pending status
                self.progress.mark_step_status(CloudProcessStep.UPLOAD_TRAINING_DATA, CloudStatus.PENDING)
                self.progress.mark_step_status(CloudProcessStep.CREATE_FINE_TUNE_JOB, CloudStatus.PENDING)
                self.progress.mark_step_status(CloudProcessStep.WAIT_FOR_FINE_TUNE_COMPLETION, CloudStatus.PENDING)

                # Save the progress
                self.progress.save_progress()
                logger.info("Cloud process steps have been set to pending status")

            self.progress.progress.data["status"] = CloudStatus.SUSPENDED
            self.progress.save_progress()

            logger.info("Cloud training process has been stopped successfully")
            return 'success'

        except Exception as e:
            logger.error(f"Error stopping cloud process: {str(e)}", exc_info=True)
            return 'failed'

import os
import re
import time
import psutil
from typing import Optional, Dict
from lpm_kernel.L1.utils import save_true_topics
from lpm_kernel.L1.serializers import NotesStorage
from lpm_kernel.kernel.note_service import NoteService
from lpm_kernel.L2.l2_generator import L2Generator
from lpm_kernel.L2.utils import save_hf_model
from lpm_kernel.api.common.responses import APIResponse
from lpm_kernel.api.domains.loads.services import LoadService
from lpm_kernel.kernel.chunk_service import ChunkService
from lpm_kernel.kernel.l1.l1_manager import (
    extract_notes_from_documents,
    document_service,
    get_latest_status_bio,
    get_latest_global_bio,
)
from lpm_kernel.api.common.script_executor import ScriptExecutor
from lpm_kernel.configs.config import Config
from lpm_kernel.file_data.chunker import DocumentChunker
from lpm_kernel.kernel.l1.l1_manager import generate_l1_from_l0
import threading
from lpm_kernel.api.domains.trainprocess.progress_enum import Status
from lpm_kernel.api.domains.trainprocess.process_step import ProcessStep
from lpm_kernel.api.domains.trainprocess.progress_holder import TrainProgressHolder
from lpm_kernel.api.domains.trainprocess.training_params_manager import TrainingParamsManager
from lpm_kernel.models.l1 import L1Bio, L1Shade
from lpm_kernel.common.repository.database_session import DatabaseSession
from lpm_kernel.api.domains.kernel.routes import store_l1_data
from lpm_kernel.api.domains.trainprocess.L1_exposure_manager import output_files, query_l1_version_data, read_file_content
from lpm_kernel.services.email_service import EmailService
import gc
import subprocess
from lpm_kernel.configs.logging import get_train_process_logger, TRAIN_LOG_FILE
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
            
            # Initialize L2 data dictionary
            self.l2_data = {
                "notes": None,
                "basic_info": None,
                "data_output_base_dir": None,
                "topics_path": None,
                "entitys_path": None,
                "graph_path": None,
                "config_path": None
            }
            self.l2_data_prepared = False
            
            # Initialize email service
            self.email_service = EmailService()
        
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
            documents = self.list_documents() 
            for doc in documents:
                doc_id = doc.get("id")

                # Directly call document service instead of API
                embedding = document_service.process_document_embedding(doc_id)
                if embedding is None:
                    logger.error(
                        f"Generate document embeddings failed for doc_id: {doc_id}"
                    )
                    self.progress.mark_step_status(ProcessStep.GENERATE_DOCUMENT_EMBEDDINGS, Status.FAILED)
                    return False
                self.progress.mark_step_status(ProcessStep.GENERATE_DOCUMENT_EMBEDDINGS, Status.COMPLETED)
                logger.info(f"Successfully generated embedding for document {doc_id}") 
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
            processed, failed = 0, 0

            chunk_service = ChunkService()
            for doc in documents:
                try:
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
            logger.info("Generating L0 data...")
            analyzed_docs = document_service.analyze_all_documents()
            logger.info(f"Successfully analyzed {len(analyzed_docs)} documents for L0")
            
            # Mark step as completed
            self.progress.mark_step_status(ProcessStep.EXTRACT_DIMENSIONAL_TOPICS, Status.COMPLETED)
            logger.info("Dimensional topics extraction (L0) completed successfully")
            return True

        except Exception as e:
            logger.error(f"Extract dimensional topics (L0) failed: {str(e)}")
            self.progress.mark_step_status(ProcessStep.EXTRACT_DIMENSIONAL_TOPICS, Status.FAILED)
            return False
            
    def generate_biography(self) -> bool:
        """Generate biography using L1 data"""
        try:
            # Mark step as in progress
            self.progress.mark_step_status(ProcessStep.GENERATE_BIOGRAPHY, Status.IN_PROGRESS)
            logger.info("Starting biography generation...")

            # Generate L1 data and biography
            logger.info("Generating L1 data and biography...")
            l1_data = generate_l1_from_l0()
            logger.info("Successfully generated L1 data and biography")

            # Store L1 data
            with DatabaseSession.session() as session:
                store_l1_data(session, l1_data)

            # Mark step as completed
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
                error_msg = f"Model path does not exist after download: {model_path}"
                logger.error(error_msg)
                self.progress.mark_step_status(ProcessStep.MODEL_DOWNLOAD, Status.FAILED)
                
                # Send email notification for model download failure
                try:
                    self.email_service.send_training_error_notification(
                        model_name=self.model_name,
                        error_message=error_msg,
                        step_context="Model Download - Path Verification"
                    )
                except Exception as email_error:
                    logger.error(f"Failed to send error notification email: {email_error}")
                
                return False

        except Exception as e:
            error_msg = f"Download model failed: {str(e)}"
            logger.error(error_msg)
            self.progress.mark_step_status(ProcessStep.MODEL_DOWNLOAD, Status.FAILED)
            
            # Send email notification for model download exception
            try:
                self.email_service.send_training_error_notification(
                    model_name=self.model_name,
                    error_message=error_msg,
                    step_context="Model Download - Exception"
                )
            except Exception as email_error:
                logger.error(f"Failed to send error notification email: {email_error}")
            
            return False

    def map_your_entity_network(self)->bool:
        """Map entity network using notes and basic info"""
        try:
            # Mark step as in progress
            self.progress.mark_step_status(ProcessStep.MAP_ENTITY_NETWORK, Status.IN_PROGRESS)
            logger.info("Starting entity network mapping...")
        
            # Get or prepare L2 data
            self._prepare_l2_data()

            l2_generator = L2Generator(
                data_path=os.path.join(os.getcwd(), "resources")
            )
            l2_generator.data_preprocess(self.l2_data["notes"], self.l2_data["basic_info"])
            
            self.progress.mark_step_status(ProcessStep.MAP_ENTITY_NETWORK, Status.COMPLETED)
            logger.info("Entity network mapping completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Map entity network failed: {str(e)}")
            self.progress.mark_step_status(ProcessStep.MAP_ENTITY_NETWORK, Status.FAILED)
            self._cleanup_resources()
            return False

    def decode_preference_patterns(self)->bool:
        """Decode preference patterns using notes and related data"""
        try:
            params_manager = TrainingParamsManager()
            training_params = params_manager.get_latest_training_params()
            concurrency_threads = training_params.get("concurrency_threads")
            data_synthesis_mode = training_params.get("data_synthesis_mode")
            os.environ["CONCURRENCY_THREADS"] = str(concurrency_threads)
            os.environ["DATA_SYNTHESIS_MODE"] = data_synthesis_mode
            
            # Mark step as in progress
            self.progress.mark_step_status(ProcessStep.DECODE_PREFERENCE_PATTERNS, Status.IN_PROGRESS)
            logger.info("Starting preference patterns decoding...")
            # Get or prepare L2 data
            self._prepare_l2_data()

            # Use data from l2_data dictionary
            training_params = TrainingParamsManager.get_latest_training_params()
            L2Generator(is_cot=training_params.get("is_cot", False)).gen_preference_data(                
                    self.l2_data["notes"],
                    self.l2_data["basic_info"],
                    self.l2_data["data_output_base_dir"],
                    self.l2_data["topics_path"],
                    self.l2_data["entitys_path"],
                    self.l2_data["graph_path"],
                    self.l2_data["config_path"]
                    )
            
            self.progress.mark_step_status(ProcessStep.DECODE_PREFERENCE_PATTERNS, Status.COMPLETED)
            logger.info("Preference patterns decoding completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Decode preference patterns failed: {str(e)}")
            self.progress.mark_step_status(ProcessStep.DECODE_PREFERENCE_PATTERNS, Status.FAILED)
            return False

    def reinforce_identity(self)->bool:
        """Reinforce identity using notes and related data"""
        try:
            # Mark step as in progress
            self.progress.mark_step_status(ProcessStep.REINFORCE_IDENTITY, Status.IN_PROGRESS)
            logger.info("Starting identity reinforcement...")
            # Get or prepare L2 data
            self._prepare_l2_data()

            # Get training parameters
            training_params = TrainingParamsManager.get_latest_training_params()
            # Use data from l2_data dictionary
            l2_generator = L2Generator(
                data_path=os.path.join(os.getcwd(), "resources"), is_cot=training_params.get("is_cot", False)
                )  
            l2_generator.gen_selfqa_data(
                    self.l2_data["notes"],
                    self.l2_data["basic_info"],
                    self.l2_data["data_output_base_dir"],
                    self.l2_data["topics_path"],
                    self.l2_data["entitys_path"],
                    self.l2_data["graph_path"],
                    self.l2_data["config_path"]
                    )
            
            self.progress.mark_step_status(ProcessStep.REINFORCE_IDENTITY, Status.COMPLETED)
            logger.info("Identity reinforcement completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Reinforce identity failed: {str(e)}")
            self.progress.mark_step_status(ProcessStep.REINFORCE_IDENTITY, Status.FAILED)
            return False
            
    def _cleanup_resources(self):
        """Clean up resources to prevent memory leaks"""
        logger.info("Cleaning up resources to prevent memory leaks")
        
        # Clean up large data structures in l2_data dictionary
        for key in self.l2_data:
            self.l2_data[key] = None
        
        self.l2_data_prepared = False
        
        # Force garbage collection
        gc.collect()
        
        # Log memory usage after cleanup
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        logger.info(f"Memory usage after cleanup: {memory_info.rss / 1024 / 1024:.2f} MB")
    
    def augment_content_retention(self) -> bool:
        """Augment content retention using notes, basic info and graph data"""
        try:
            # Mark step as in progress
            self.progress.mark_step_status(ProcessStep.AUGMENT_CONTENT_RETENTION, Status.IN_PROGRESS)
            logger.info("Starting content retention augmentation...")
            # Get or prepare L2 data
            self._prepare_l2_data()

            # Get training parameters
            training_params = TrainingParamsManager.get_latest_training_params()
            # Use data from l2_data dictionary
            l2_generator = L2Generator(data_path=os.path.join(os.getcwd(), "resources"), is_cot=training_params.get("is_cot", False))
            l2_generator.gen_diversity_data(
                self.l2_data["notes"],
                self.l2_data["basic_info"],
                self.l2_data["data_output_base_dir"],
                self.l2_data["topics_path"],
                self.l2_data["entitys_path"],
                self.l2_data["graph_path"],
                self.l2_data["config_path"]
            )
            l2_generator.merge_json_files(self.l2_data["data_output_base_dir"])
            # Mark step as completed
            logger.info("Content retention augmentation completed successfully")
            self.progress.mark_step_status(ProcessStep.AUGMENT_CONTENT_RETENTION, Status.COMPLETED)
            
            # Clean up resources after completion
            self._cleanup_resources()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to augment content retention: {str(e)}")
            self.progress.mark_step_status(ProcessStep.AUGMENT_CONTENT_RETENTION, Status.FAILED)
            # Clean up resources even if there was an error
            self._cleanup_resources()
            return False

    def _prepare_l2_data(self) -> dict:
        """Prepare common data needed for L2 generation tasks using lazy loading

        Returns:
            Dictionary containing all L2 data:
            - notes: List of prepared notes
            - basic_info: Dict containing user information
            - data_output_base_dir: Path to output directory
            - topics_path: Path to topics data
            - entitys_path: Path to entity mapping file
            - graph_path: Path to graph data
            - config_path: Path to config file
        """
        # If data is already prepared, return cached data directly
        if self.l2_data_prepared and all(self.l2_data.values()):
            logger.info("Using cached L2 data")
            return self.l2_data

        logger.info("Preparing L2 data...")

        # Setup directories and paths
        config = Config.from_env()
        base_dir = os.path.join(
            os.getcwd(), config.get("USER_DATA_PIPELINE_DIR") + "/raw_data"
        )
        os.makedirs(base_dir, exist_ok=True)

        # get topic
        topics_path = os.path.join(base_dir, "topics.json")
        self.l2_data["topics_path"] = topics_path
        logger.info("Topics data not found, generating it...")
        chunk_service = ChunkService()
        topics_data = chunk_service.query_topics_data()
        save_true_topics(topics_data, topics_path)

        # Initialize storage
        storage = NotesStorage()
        logger.info("Notes not found, preparing them...")
        documents = document_service.list_documents_with_l0()
        logger.info(f"list_documents_with_l0 len: {len(documents)}")
        notes_list, _ = extract_notes_from_documents(documents)
        logger.info(f"extract_notes_from_documents len: {len(notes_list)}")
        note_service = NoteService()
        note_service.prepareNotes(notes_list)
        storage.save_notes(notes_list)
        self.l2_data["notes"] = storage.load_notes()

        # Get paths
        self.l2_data["config_path"] = os.path.join(
            os.getcwd(),
            "resources/L2/data_pipeline/data_prep/subjective/config/config.json",
        )
        self.l2_data["entitys_path"] = os.path.join(
            os.getcwd(),
            "resources/L2/data_pipeline/raw_data/id_entity_mapping_subjective_v2.json",
        )
        self.l2_data["graph_path"] = os.path.join(
            os.getcwd(),
            "resources/L1/graphrag_indexing_output/subjective/entities.parquet",
        )
        self.l2_data["data_output_base_dir"] = os.path.join(os.getcwd(), "resources/L2/data")

        # Lazy load user information
        logger.info("Loading user information...")
        status_bio = get_latest_status_bio()
        global_bio = get_latest_global_bio()
        self.l2_data["basic_info"] = {
            "username": LoadService.get_current_upload_name(),
            "aboutMe": LoadService.get_current_upload_description(),
            "statusBio": status_bio.content if status_bio else "Currently working on an AI project.",
            "globalBio": global_bio.content_third_view if global_bio
                else "The User is a software engineer who loves programming and learning new technologies.",
            "lang": "English",
        }

        # Mark data as prepared
        self.l2_data_prepared = True

        return self.l2_data
    def train(self) -> bool:
        """Start model training"""
        try:
            # Mark step as in progress
            self.progress.mark_step_status(ProcessStep.TRAIN, Status.IN_PROGRESS)
            
            # Send training start notification
            notification_email = self.config.get('NOTIFICATION_EMAIL')
            if notification_email:
                try:
                    # Get training parameters for notification
                    params_manager = TrainingParamsManager()
                    training_params = params_manager.get_latest_training_params()
                    
                    # Collect system information
                    system_info = {}
                    try:
                        process = psutil.Process(os.getpid())
                        memory_info = process.memory_info()
                        system_info = {
                            "Memory Usage": f"{memory_info.rss / 1024 / 1024:.2f} MB",
                            "CPU Usage": f"{process.cpu_percent(interval=0.1):.1f}%",
                            "Process ID": os.getpid(),
                            "Timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                        }
                    except Exception as e:
                        logger.warning(f"Could not collect system info for start notification: {str(e)}")
                    
                    # Send start notification with training parameters
                    self.email_service.send_training_start_notification(
                        self.model_name,
                        notification_email,
                        training_params,
                        system_info=system_info
                    )
                    logger.info(f"Training start notification sent to {notification_email}")
                except Exception as e:
                    logger.warning(f"Could not send training start notification: {str(e)}")
                    # Continue with training even if notification fails
            
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
                error_msg = "Training process failed to start"
                logger.error(error_msg)
                self.progress.mark_step_status(ProcessStep.TRAIN, Status.FAILED)
                # Send email notification for training failure
                notification_email = self.config.get('NOTIFICATION_EMAIL')
                if notification_email:
                    self.email_service.send_training_error_notification(
                        self.model_name,
                        error_msg,
                        notification_email
                    )
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
                    # Send email notification for training failure with enhanced error context
                    notification_email = self.config.get('NOTIFICATION_EMAIL')
                    if notification_email:
                        # Collect error context information
                        error_context = {
                            "Process Step": "TRAIN",
                            "Model Name": self.model_name,
                            "Timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "Process ID": self.current_pid if hasattr(self, 'current_pid') else "N/A",
                            "Memory Usage": f"{psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024:.2f} MB"
                        }
                        
                        # Add training parameters to error context if available
                        try:
                            params_manager = TrainingParamsManager()
                            training_params = params_manager.get_latest_training_params()
                            if training_params:
                                error_context.update({
                                    "Learning Rate": training_params.get("learning_rate", "N/A"),
                                    "Epochs": training_params.get("number_of_epochs", "N/A"),
                                    "Concurrency Threads": training_params.get("concurrency_threads", "N/A"),
                                    "Data Synthesis Mode": training_params.get("data_synthesis_mode", "N/A"),
                                    "Use CUDA": training_params.get("use_cuda", False)
                                })
                        except Exception as e:
                            logger.warning(f"Could not add training parameters to error context: {str(e)}")
                        
                        # Send enhanced error notification
                        self.email_service.send_training_error_notification(
                            self.model_name,
                            error_msg,
                            notification_email,
                            log_file_path=TRAIN_LOG_FILE if os.path.exists(TRAIN_LOG_FILE) else None,
                            error_context=error_context,
                            max_log_lines=100  # Include more log lines for better context
                        )
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
            env["FORCE_COLOR"] = "1"       # Force colored output
            env["TQDM_FORCE_TTY"] = "1"    # Force tqdm to use TTY features
            
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
                        # Send progress notification at key milestones (25%, 50%, 75%)
                        notification_email = self.config.get('NOTIFICATION_EMAIL')
                        if notification_email and hasattr(self, 'last_notified_percentage'):
                            # Check if we've crossed a notification threshold
                            thresholds = [25.0, 50.0, 75.0]
                            for threshold in thresholds:
                                if self.last_notified_percentage < threshold and percentage >= threshold:
                                    try:
                                        # Calculate estimated time remaining
                                        elapsed_time = time.time() - self.training_start_time
                                        if percentage > 0:
                                            estimated_total_time = elapsed_time / (percentage / 100.0)
                                            estimated_remaining = estimated_total_time - elapsed_time
                                            time_remaining_str = f"{estimated_remaining / 60:.1f} minutes" if estimated_remaining < 3600 else f"{estimated_remaining / 3600:.1f} hours"
                                        else:
                                            time_remaining_str = "Unknown"
                                        
                                        # Collect progress statistics
                                        progress_stats = {
                                            "Percentage Complete": f"{percentage:.1f}%",
                                            "Current Step": current_step,
                                            "Total Steps": total_steps,
                                            "Elapsed Time": f"{elapsed_time / 60:.1f} minutes" if elapsed_time < 3600 else f"{elapsed_time / 3600:.1f} hours",
                                            "Estimated Time Remaining": time_remaining_str
                                        }
                                        
                                        # Send progress notification
                                        self.email_service.send_training_progress_notification(
                                            self.model_name,
                                            notification_email,
                                            int(percentage),
                                            progress_stats
                                        )
                                        logger.info(f"Training progress notification sent at {percentage:.1f}%")
                                        self.last_notified_percentage = percentage
                                    except Exception as e:
                                        logger.warning(f"Could not send progress notification: {str(e)}")
                        
                        if percentage == 100.0:
                            self.progress.mark_step_status(ProcessStep.TRAIN, Status.COMPLETED)
                            # Send email notification for training completion with enhanced information
                            notification_email = self.config.get('NOTIFICATION_EMAIL')
                            if notification_email:
                                # Calculate training duration
                                training_duration = time.time() - last_update_time
                                duration_str = f"{training_duration / 60:.2f} minutes" if training_duration < 3600 else f"{training_duration / 3600:.2f} hours"
                                
                                # Get model paths
                                model_paths = self._get_model_paths(self.model_name)
                                model_path_str = model_paths.get("merged_dir", "Unknown")
                                
                                # Collect comprehensive training statistics
                                training_stats = {
                                    "Total Steps": total_steps,
                                    "Completed Steps": current_step,
                                    "Training Duration": duration_str,
                                    "Average Time Per Step": f"{training_duration / max(current_step, 1):.2f} seconds"
                                }
                                
                                # Add system resource information
                                try:
                                    process = psutil.Process(os.getpid())
                                    memory_info = process.memory_info()
                                    training_stats.update({
                                        "Peak Memory Usage": f"{memory_info.rss / 1024 / 1024:.2f} MB",
                                        "CPU Usage": f"{process.cpu_percent(interval=0.1):.1f}%"
                                    })
                                except Exception as e:
                                    logger.warning(f"Could not add system resource info to training stats: {str(e)}")
                                
                                # Add training parameters to statistics
                                try:
                                    params_manager = TrainingParamsManager()
                                    training_params = params_manager.get_latest_training_params()
                                    if training_params:
                                        training_stats.update({
                                            "Learning Rate": training_params.get("learning_rate", "N/A"),
                                            "Epochs": training_params.get("number_of_epochs", "N/A"),
                                            "Data Synthesis Mode": training_params.get("data_synthesis_mode", "N/A")
                                        })
                                except Exception as e:
                                    logger.warning(f"Could not add training parameters to stats: {str(e)}")
                                
                                # Send enhanced completion notification
                                self.email_service.send_training_completion_notification(
                                    self.model_name,
                                    notification_email,
                                    training_stats,
                                    model_path=model_path_str,
                                    training_duration=duration_str
                                )
                            return True
                        
                        self._update_progress("training_to_create_second_me", "train", int(percentage / 3), f"Current step: {current_step}/{total_steps}")
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
                step,   # step
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
                                    completed_files_size = sum([file_sizes.get(f, 0) for f in file_sizes if f != file_name])
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
                error_msg = f"Model '{self.model_name}' does not exist, please download first"
                logger.error(error_msg)
                self.progress.mark_step_status(ProcessStep.MERGE_WEIGHTS, Status.FAILED)
                # Send email notification for missing base model
                notification_email = self.config.get('NOTIFICATION_EMAIL')
                if notification_email:
                    error_context = {
                        "Process Step": "MERGE_WEIGHTS",
                        "Model Name": self.model_name,
                        "Timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "Base Model Path": paths["base_path"]
                    }
                    self.email_service.send_training_error_notification(
                        self.model_name,
                        error_msg,
                        notification_email,
                        error_context=error_context
                    )
                return False
            
            # Check if training output exists
            if not os.path.exists(paths["personal_dir"]):
                error_msg = f"Model '{self.model_name}' training output does not exist, please train model first"
                logger.error(error_msg)
                self.progress.mark_step_status(ProcessStep.MERGE_WEIGHTS, Status.FAILED)
                # Send email notification for missing training output
                notification_email = self.config.get('NOTIFICATION_EMAIL')
                if notification_email:
                    error_context = {
                        "Process Step": "MERGE_WEIGHTS",
                        "Model Name": self.model_name,
                        "Timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "Personal Training Path": paths["personal_dir"]
                    }
                    self.email_service.send_training_error_notification(
                        self.model_name,
                        error_msg,
                        notification_email,
                        error_context=error_context
                    )
                return False

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
                # Send email notification for script execution failure
                notification_email = self.config.get('NOTIFICATION_EMAIL')
                if notification_email:
                    error_context = {
                        "Process Step": "MERGE_WEIGHTS",
                        "Model Name": self.model_name,
                        "Timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "Script Path": script_path,
                        "Return Code": result.get('returncode', 'N/A')
                    }
                    self.email_service.send_training_error_notification(
                        self.model_name,
                        error_msg,
                        notification_email,
                        log_file_path=log_path if os.path.exists(log_path) else None,
                        error_context=error_context
                    )
                return False
                
            # Check if merged model files exist
            config_path = os.path.join(paths["merged_dir"], "config.json")
            if not os.path.exists(config_path):
                error_msg = f"Merged model files not found in {paths['merged_dir']}"
                logger.error(error_msg)
                self.progress.mark_step_status(ProcessStep.MERGE_WEIGHTS, Status.FAILED)
                # Send email notification for missing merged files
                notification_email = self.config.get('NOTIFICATION_EMAIL')
                if notification_email:
                    error_context = {
                        "Process Step": "MERGE_WEIGHTS",
                        "Model Name": self.model_name,
                        "Timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "Expected Config Path": config_path,
                        "Merged Directory": paths["merged_dir"]
                    }
                    self.email_service.send_training_error_notification(
                        self.model_name,
                        error_msg,
                        notification_email,
                        log_file_path=log_path if os.path.exists(log_path) else None,
                        error_context=error_context
                    )
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
                error_msg = f"Model '{self.model_name}' merged output does not exist, please merge model first"
                logger.error(error_msg)
                self.progress.mark_step_status(ProcessStep.CONVERT_MODEL, Status.FAILED)
                # Send email notification for missing merged model
                notification_email = self.config.get('NOTIFICATION_EMAIL')
                if notification_email:
                    error_context = {
                        "Process Step": "CONVERT_MODEL",
                        "Model Name": self.model_name,
                        "Timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "Expected Merged Path": merged_model_dir
                    }
                    self.email_service.send_training_error_notification(
                        self.model_name,
                        error_msg,
                        notification_email,
                        error_context=error_context
                    )
                return False
            
            # Get GGUF output directory
            gguf_dir = paths["gguf_dir"]
            logger.info(f"GGUF output directory: {gguf_dir}")
            
            script_path = os.path.join(os.getcwd(), "lpm_kernel/L2/convert_hf_to_gguf.py")
            gguf_path = os.path.join(gguf_dir, "model.gguf")
            logger.info(f"GGUF output path: {gguf_path}")
            
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
            
            logger.info(f"Model conversion result: {result}")
            
            # Check if script execution was successful
            if result.get('returncode', 1) != 0:
                error_msg = f"Model conversion failed: {result.get('error', 'Unknown error')}"
                logger.error(error_msg)
                self.progress.mark_step_status(ProcessStep.CONVERT_MODEL, Status.FAILED)
                # Send email notification for conversion script failure
                notification_email = self.config.get('NOTIFICATION_EMAIL')
                if notification_email:
                    error_context = {
                        "Process Step": "CONVERT_MODEL",
                        "Model Name": self.model_name,
                        "Timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "Script Path": script_path,
                        "Return Code": result.get('returncode', 'N/A'),
                        "GGUF Output Path": gguf_path
                    }
                    self.email_service.send_training_error_notification(
                        self.model_name,
                        error_msg,
                        notification_email,
                        error_context=error_context
                    )
                return False
                
            # Check if GGUF model file exists
            if not os.path.exists(gguf_path):
                error_msg = f"GGUF model file not found at {gguf_path}"
                logger.error(error_msg)
                self.progress.mark_step_status(ProcessStep.CONVERT_MODEL, Status.FAILED)
                
                # Send email notification for missing GGUF file
                try:
                    self.email_service.send_training_error_notification(
                        model_name=self.model_name,
                        error_message=error_msg,
                        step_context="Model Conversion - GGUF File Check"
                    )
                except Exception as email_error:
                    logger.error(f"Failed to send error notification email: {email_error}")
                
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
                current_step = self.progress.progress.stages[self.progress.progress.current_stage].current_step
                if current_step:
                    step = ProcessStep(current_step)
                    self.progress.mark_step_status(step, Status.FAILED)
            return False

    def start_process(self) -> bool:
        """Start training process"""
        try:
            self.is_stopped = False
            # Store the current process PID
            self.current_pid = os.getpid()  # Store the PID
            logger.info(f"Training process started with PID: {self.current_pid}")
            # Get the ordered list of all steps
            ordered_steps = ProcessStep.get_ordered_steps()

            # Get the last successfully completed step
            last_successful_step = self.progress.get_last_successful_step()
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
            if step_name == "generate_biography":
                logger.info("Querying L1 version data for biography")
                return query_l1_version_data(1)

            # If step_name is not provided or invalid, return None
            if not step_name or step_name not in output_files:
                return None
            
            # Get file path for the requested step
            file_path = output_files[step_name]
            if not os.path.exists(file_path):
                return None
            
            # Read and return file content
            return read_file_content(file_path)
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
                    current_stage = next((s for s in self.progress.progress.data["stages"] if s["name"] == current_stage_name), None)
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
                    
                    # Note: We don't terminate the main process as it's this process
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
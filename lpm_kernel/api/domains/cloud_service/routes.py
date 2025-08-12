from flask import Blueprint, jsonify, request, Response
from typing import Dict, Any, Optional, List

from sqlalchemy import table
from ...common.responses import APIResponse
from ...common.errors import APIError, ErrorCodes
import logging
import gc
import os
import time
import json
import psutil
import tempfile
import threading
import signal
import sys
from multiprocessing import Process
from datetime import datetime
from pathlib import Path
from ....configs.config import Config
from .service import CloudService
from .cloud_trainprocess_service import CloudTrainProcessService
from .cloud_progress_holder import CloudProgressHolder, CloudStatus
from ...services.user_llm_config_service import UserLLMConfigService
from ...dto.user_llm_config_dto import UpdateUserLLMConfigDTO
from lpm_kernel.api.domains.cloud_service.dto.cloud_inference_dto import CloudInferenceRequest
from lpm_kernel.api.services.local_llm_service import local_llm_service

from lpm_kernel.configs.logging import get_train_process_logger
logger = get_train_process_logger()
cloud_bp = Blueprint("cloud_service", __name__, url_prefix="/api/cloud_service")

# Global variables for tracking cloud service status
_cloud_service_active = False
_cloud_service_model_id = None

def get_service_status_file_path():
    """Get the path for service status file"""
    return os.path.join(os.getcwd(), "data", "service_status.json")

def create_service_status_file(service_type: str, model_data: dict):
    """Create service status file to track active service
    
    Args:
        service_type: 'local' or 'cloud'
        model_data: Dictionary containing model information
    """
    status_data = {
        "service_type": service_type,
        "model_data": model_data,
        "created_at": datetime.now().isoformat(),
        "status": "active"
    }
    
    status_file_path = get_service_status_file_path()
    os.makedirs(os.path.dirname(status_file_path), exist_ok=True)
    
    with open(status_file_path, 'w', encoding='utf-8') as f:
        json.dump(status_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Service status file created: {status_file_path}")

def remove_service_status_file():
    """Remove service status file when service is stopped"""
    status_file_path = get_service_status_file_path()
    if os.path.exists(status_file_path):
        os.remove(status_file_path)
        logger.info(f"Service status file removed: {status_file_path}")

def get_service_status():
    """Get current service status from file"""
    status_file_path = get_service_status_file_path()
    if not os.path.exists(status_file_path):
        return None
    try:
        with open(status_file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to read service status file: {str(e)}")
        return None

@cloud_bp.route("/set_api_key", methods=["POST"])
def set_api_key():
    """
    Request: JSON object, containing:
    - api_key: str, Cloud Service API Key
    """
    try:
        data = request.json
        api_key = data.get('api_key')
        
        if not api_key:
            return jsonify(APIResponse.error("API key is required"))
        
        user_llm_config_service = UserLLMConfigService()
        
        config = user_llm_config_service.get_available_llm()
        
        update_data = {}
        if config:
            update_data = config.dict()
        
        update_data['cloud_service_api_key'] = api_key
        
        dto = UpdateUserLLMConfigDTO(**update_data)
        user_llm_config_service.update_config(1, dto)
            
        return jsonify(APIResponse.success(message="API key setting successful and saved to database"))
        
    except Exception as e:
        logger.error(f"Failed to set API key: {str(e)}", exc_info=True)
        return jsonify(APIResponse.error(f"Failed to set API key: {str(e)}"))


@cloud_bp.route("/get_api_key", methods=["GET"])
def get_api_key():
    try:
        user_llm_config_service = UserLLMConfigService()
        config = user_llm_config_service.get_available_llm()
        
        api_key = ""
        if config and hasattr(config, 'cloud_service_api_key'):
            api_key = config.cloud_service_api_key or ""
            
            if api_key:
                return jsonify(APIResponse.success(data={
                    "api_key": api_key
                }))
        
        return jsonify(APIResponse.success(data={
            "api_key": ""
        }))
    except Exception as e:
        logger.error(f"Failed to get API key: {str(e)}", exc_info=True)
        return jsonify(APIResponse.error(f"Failed to get API key: {str(e)}"))

@cloud_bp.route("/list_available_models", methods=["GET"])
def list_available_models():
    try:
        cloud_service = CloudService()
        
        models = cloud_service.list_available_models()
        
        if not models:
            logger.warning("No models available for fine-tuning")
        
        return jsonify(APIResponse.success(data=models))
    
    except Exception as e:
        logger.error(f"Failed to list available models: {str(e)}", exc_info=True)
        return jsonify(APIResponse.error(f"Failed to list available models: {str(e)}"))


# ============= Cloud Training Process Routes =============

@cloud_bp.route("/train/resume", methods=["POST"])
def resume_cloud_training():
    """Resume cloud training process from the last checkpoint
    
    Request: JSON object, containing:
    - model_name: str, optional, the model name to resume training for
    - base_model: str, optional, the base model to use for fine-tuning
    - training_type: str, optional, the training type to use
    - hyper_parameters: dict, optional, hyperparameters for training
    
    If model_name is not provided, the system will attempt to resume the most recent training process.
    """
    try:

        cloud_train_service = CloudTrainProcessService.get_instance()

        if cloud_train_service is None:
            logger.warning("No training parameters found in file")
            return jsonify(APIResponse.error("No training parameters found. Please use /train/start endpoint for initial training."))

        if cloud_train_service.progress and cloud_train_service.progress.progress and cloud_train_service.progress.progress.data:
            cloud_train_service.progress.progress.data["status"] = CloudStatus.IN_PROGRESS
            cloud_train_service.progress.save_progress()
            logger.info("Reset cloud training status from suspended to in_progress")

        thread = threading.Thread(target=cloud_train_service.start_process)
        thread.daemon = True
        thread.start()

        return jsonify(APIResponse.success("Cloud resume training process started successfully"))
    
    except Exception as e:
        logger.error(f"Failed to resume cloud training: {str(e)}", exc_info=True)
        return jsonify(APIResponse.error(f"Failed to resume cloud training: {str(e)}"))

@cloud_bp.route("/train/stop", methods=["POST"])
def stop_cloud_training():
    """Stop cloud training process
    
    Request: JSON object, containing:
    - model_name: str, optional, the model name to stop training for
    
    If model_name is not provided, the system will attempt to stop the most recent training process.
    """
    try:

        train_service = CloudTrainProcessService.get_instance()
        
        if not train_service:
            return jsonify(APIResponse.error("No training parameters found. Please use /train/start endpoint for initial training."))

        result = train_service.stop_process()
        
        if result == 'success':
            return jsonify(APIResponse.success(message=f"Cloud training process stopped successfully", data={"status": "success"}))
        elif result == 'pending':
            return jsonify(APIResponse.success(message=f"Cloud training process is in the process of stopping", data={"status": "pending"}))
        else:  # 'failed'
            return jsonify(APIResponse.error(message=f"Failed to stop cloud training process", data={"status": "failed"}))
    
    except Exception as e:
        logger.error(f"Failed to stop cloud training process: {str(e)}", exc_info=True)
        return jsonify(APIResponse.error(f"Failed to stop cloud training process: {str(e)}"))

@cloud_bp.route("/train/start", methods=["POST"])
def start_cloud_training():
    """Start cloud training process"""
    try:
        params_dir = Path("data/cloud_progress")
        if params_dir.exists():
            
            logger.info("Cleaning cloud_progress directory...")
            for file_path in params_dir.glob("*"):
                if file_path.is_file():
                    os.remove(file_path)
                    logger.info(f"Removed file: {file_path}")
            logger.info("Cloud_progress directory cleaned successfully")
            
        data = request.json
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        model_name = timestamp
        
        base_model = data.get("base_model")
        training_type = data.get("training_type", "efficient_sft")
        hyper_parameters = data.get("hyper_parameters", {})
        data_synthesis_mode = data.get("data_synthesis_mode", "low")
        language = data.get("language", "en")
        
        # Data filtering parameters
        data_filtering_model = data.get("data_filtering_model", "gemma:2b")
        data_filtering_workers = data.get("data_filtering_workers", 5)
        data_filtering_keep_ratio = data.get("data_filtering_keep_ratio", 0.8)

        os.environ["DATA_SYNTHESIS_MODE"] = data_synthesis_mode
        
        training_params = {
            "model_name": model_name,
            "base_model": base_model,
            "training_type": training_type,
            "hyper_parameters": hyper_parameters,
            "data_synthesis_mode": data_synthesis_mode,
            "language": language,
            "data_filtering_model": data_filtering_model,
            "data_filtering_workers": data_filtering_workers,
            "data_filtering_keep_ratio": data_filtering_keep_ratio,
            "created_at": datetime.now().isoformat()
        }
        

        params_dir = Path("data/cloud_progress")
        params_dir.mkdir(parents=True, exist_ok=True)
        
        params_file = params_dir / "cloud_training_params.json"
        with open(params_file, "w", encoding="utf-8") as f:
            json.dump(training_params, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Training parameters saved to {params_file}")
        
        train_service = CloudTrainProcessService(current_model_name=model_name, base_model=base_model, training_type=training_type, hyper_parameters=hyper_parameters)
        
        def async_train_process():
            try:
                logger.info(f"Starting async cloud training process for model: {model_name}")
                success = train_service.start_process()
                if not success:
                    # Check if it was stopped by user
                    if train_service.is_stopped:
                        logger.info("Async cloud training process was stopped by user")
                    else:
                        logger.error("Async cloud training process failed")
                else:
                    logger.info(f"Async cloud training process completed successfully with job_id: {train_service.job_id}")
            except Exception as e:
                logger.error(f"Async cloud training process failed with error: {str(e)}", exc_info=True)
        
        thread = threading.Thread(target=async_train_process, daemon=True)
        thread.start()
        logger.info(f"Started async thread for cloud training process")
        
        return jsonify(APIResponse.success("Cloud training process started successfully"))
    except Exception as e:
        logger.error(f"Start cloud training failed: {str(e)}", exc_info=True)
        return jsonify(APIResponse.error(f"Failed to start cloud training: {str(e)}"))


@cloud_bp.route("/train/status/training/<job_id>", methods=["GET"])
def get_cloud_training_status(job_id):
    """Get cloud training status"""
    try:
        cloud_service = CloudService()
        model_id = None
        status = cloud_service.check_fine_tune_status(job_id)

        if not status:
            return jsonify(APIResponse.error("Failed to get training status"))

        if status == "SUCCEEDED":
            model_id = cloud_service.model_id
        
        return jsonify(APIResponse.success(data=model_id, message=status))
    except Exception as e:
        logger.error(f"Get cloud training status failed: {str(e)}", exc_info=True)
        return jsonify(APIResponse.error(f"Failed to get cloud training status: {str(e)}"))

@cloud_bp.route("/train/progress", methods=["GET"])
def get_cloud_training_progress():
    """Get detailed progress information for cloud training"""
    try:
        # Directly read JSON file content
        progress_file = Path("data/cloud_progress/cloud_progress.json")
        
        if progress_file.exists():
            try:
                # Directly read file content
                with open(progress_file, "r", encoding="utf-8") as f:
                    progress_data = json.load(f)
                
                # Build response data
                response_data = {
                    "progress": progress_data,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "job_id": progress_data.get("job_id")
                }
                
                return jsonify(APIResponse.success(data=response_data))
            except Exception as e:
                logger.error(f"Error reading progress file: {str(e)}")
                return jsonify(APIResponse.error(f"Unable to read progress file: {str(e)}"))
        else:
            # If file doesn't exist, return empty progress
            logger.warning(f"Progress file doesn't exist: {progress_file}")
            empty_progress = {
                "stages": [],
                "overall_progress": 0,
                "current_stage": None,
                "status": "pending",
                "message": "No training in progress",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "model_name": None,
                "job_id": None
            }
            
            response_data = {
                "progress": empty_progress,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "job_id": None
            }
            
            return jsonify(APIResponse.success(data=response_data))
    except Exception as e:
        logger.error(f"Get cloud training progress failed: {str(e)}", exc_info=True)
        return jsonify(APIResponse.error(f"Failed to get cloud training progress: {str(e)}"))

@cloud_bp.route("/train/progress/reset", methods=["POST"])
def reset_cloud_training_progress():
    """Reset cloud training progress and force initialize cloud_progress.json file"""
    try:
        # Get current CloudTrainProcessService instance
        train_service = CloudTrainProcessService.get_instance()
        
        # If training service instance exists, terminate it immediately
        if train_service:
            logger.info(f"Found running CloudTrainProcessService instance, forcibly terminating it...")
            
            # Set stop flag
            train_service.is_stopped = True
            
            # General function to terminate process
            def kill_process_with_children(pid, process_name):
                try:
                    if not psutil.pid_exists(pid):
                        logger.warning(f"{process_name} with PID {pid} no longer exists")
                        return
                        
                    logger.info(f"Forcibly terminating {process_name} with PID: {pid}")
                    process = psutil.Process(pid)
                    
                    # Get and terminate all child processes
                    children = process.children(recursive=True)
                    for child in children:
                        try:
                            logger.info(f"Killing child process with PID: {child.pid}")
                            child.kill()  # Use kill instead of terminate to ensure immediate termination
                        except Exception as e:
                            logger.error(f"Error killing child process: {str(e)}")
                    
                    # Terminate main process
                    try:
                        process.kill()  # Use kill instead of terminate
                        logger.info(f"Successfully killed {process_name} with PID: {pid}")
                    except Exception as e:
                        logger.error(f"Error killing main process: {str(e)}")
                except Exception as e:
                    logger.error(f"Error in kill_process_with_children for {process_name}: {str(e)}", exc_info=True)
            
            # Terminate data processing process
            if hasattr(train_service, '_data_processing_process') and train_service._data_processing_process and train_service._data_processing_pid:
                kill_process_with_children(train_service._data_processing_pid, "data processing process")
                
            # Terminate wait completion process
            if hasattr(train_service, '_wait_completion_process') and train_service._wait_completion_process and train_service._wait_completion_pid:
                kill_process_with_children(train_service._wait_completion_pid, "wait completion process")
            
            # If there's a job ID, record it but don't wait for cancellation result
            job_id = None
            if hasattr(train_service, 'job_id') and train_service.job_id:
                job_id = train_service.job_id
                logger.info(f"Found running job: {job_id}, will be cancelled separately")
            
            # Force reset instance variables without waiting for any process or thread to complete
            CloudTrainProcessService._instance = None
            CloudTrainProcessService._initialized = False
            logger.info("Forcibly reset CloudTrainProcessService instance variables")
            
            # If there's a job ID, send cancellation request in the background
            if job_id:
                try:
                    # Create new CloudService instance to cancel the task, avoid using the reset train_service
                    cloud_service = CloudService()
                    
                    # Define process function to cancel the job
                    def cancel_job_process(job_id):
                        try:
                            # Register signal handler
                            def handle_sigterm(signum, frame):
                                logger.info(f"Cancel job process received SIGTERM signal, exiting...")
                                sys.exit(0)
                                
                            signal.signal(signal.SIGTERM, handle_sigterm)
                            
                            # Send cancellation request
                            cloud_service.cancel_fine_tune_job(job_id)
                        except Exception as e:
                            logger.error(f"Error in cancel job process: {str(e)}", exc_info=True)
                    
                    # Use process to send cancellation request without blocking main process
                    cancel_process = Process(
                        target=cancel_job_process,
                        args=(job_id,),
                        name="CancelJobProcess",
                        daemon=True
                    )
                    cancel_process.start()
                    logger.info(f"Started background process to cancel job {job_id} with PID: {cancel_process.pid}")
                except Exception as e:
                    logger.error(f"Error starting job cancellation thread: {str(e)}")
        
        # Clear files related to existing CloudTrainProcessService instance
        params_dir = Path("data/cloud_progress")
        params_dir.mkdir(parents=True, exist_ok=True)
        
        # Delete all related files
        files_to_delete = [
            "cloud_progress.json",         # Progress file
            "job_id.json"                 # Job ID file
        ]
        
        for file_name in files_to_delete:
            file_path = params_dir / file_name
            if file_path.exists():
                try:
                    os.remove(file_path)
                    logger.info(f"Deleted file: {file_path}")
                except Exception as e:
                    logger.error(f"Error deleting file {file_path}: {str(e)}")
        
        # Force reset CloudTrainProcessService instance variables
        CloudTrainProcessService._instance = None
        CloudTrainProcessService._initialized = False  # Reset initialization flag
        logger.info("Reset CloudTrainProcessService instance variables")
        
        # Create brand new progress holder
        new_progress_holder = CloudProgressHolder()
        
        # Reset progress
        new_progress_holder.progress.reset()
        
        gc.collect()
        logger.info("Forced garbage collection to clean up any lingering references")
        
        # Define progress file path
        progress_file_path = params_dir / "cloud_progress.json"
        
        # Save initialized progress (only save once)
        new_progress_holder.save_progress()
        logger.info(f"Created new progress file with completely fresh state: {progress_file_path}")
        
        return jsonify(APIResponse.success(message=f"Cloud training progress has been completely reset with a fresh state"))
    except Exception as e:
        logger.error(f"Reset cloud training progress failed: {str(e)}", exc_info=True)
        return jsonify(APIResponse.error(f"Failed to reset cloud training progress: {str(e)}"))

# ... (other code remains unchanged)
def search_job_info():
    try:
        # Use data/cloud_progress folder to store job_id.json
        params_dir = Path("data/cloud_progress")
        params_dir.mkdir(parents=True, exist_ok=True)
        job_file_path = params_dir / "job_id.json"
        
        if not job_file_path.exists():
            return jsonify(APIResponse.success(data={
                "exists": False,
                "message": "Job information file not found"
            }))
        
        try:
            with open(job_file_path, "r") as f:
                job_info = json.load(f)
                
            return jsonify(APIResponse.success(data=job_info))
        except json.JSONDecodeError:
            return jsonify(APIResponse.error("Invalid JSON format in job information file"))
    except Exception as e:
        logger.error(f"Get job info failed: {str(e)}", exc_info=True)
        return jsonify(APIResponse.error(f"Failed to get job information: {str(e)}"))

@cloud_bp.route("/train/deployment_status/<model_id>", methods=["GET"])
def check_cloud_deployment_status(model_id):
    """Check deployment status"""
    try:
        cloud_service = CloudService()

        status = cloud_service.check_deployment_status(model_id)
        
        if status is None:
            return jsonify(APIResponse.error("Failed to check deployment status"))
        
        return jsonify(APIResponse.success(data=status))
    except Exception as e:
        logger.error(f"Check deployment status failed: {str(e)}", exc_info=True)
        return jsonify(APIResponse.error(f"Failed to check deployment status: {str(e)}"))

@cloud_bp.route("/train/inference", methods=["POST"])
def run_cloud_inference():
    """Run inference using deployed model with local knowledge retrieval
    
    This endpoint accepts a request in OpenAI-compatible format and returns a response
    in the same format. It supports both streaming and non-streaming responses.
    It also supports local knowledge retrieval before sending to cloud inference.
    
    Request format:
    {
        "messages": [{"role": "user", "content": "Hello"}],
        "model_id": "your-model-id",
        "temperature": 0.1,
        "max_tokens": 2000,
        "stream": true,
        "enable_l0_retrieval": false,
        "enable_l1_retrieval": false,
        "role_id": "optional-role-id"
    }
    """
    try:
        
        try:
            body = CloudInferenceRequest(**request.json)
        except Exception as e:
            logger.error(f"Invalid request format: {str(e)}")
            return jsonify(APIResponse.error(f"Invalid request format: {str(e)}"))
        
        # 1. Check required parameters
        if not body.messages:
            return jsonify(APIResponse.error("messages are required"))
        if not body.model_id:
            return jsonify(APIResponse.error("model_id is required"))

        # 2. Perform local knowledge retrieval (if enabled)
        enhanced_messages = body.messages.copy()
        
        if body.enable_l0_retrieval or body.enable_l1_retrieval:
            logger.info("Performing local knowledge retrieval before cloud inference")
            
            # Get knowledge-enhanced messages from local ChatService
            from lpm_kernel.api.domains.kernel2.dto.chat_dto import ChatRequest
            from lpm_kernel.api.domains.kernel2.services.chat_service import chat_service
            
            # Create temporary ChatRequest object for knowledge retrieval
            temp_chat_request = ChatRequest(
                message="",  
                messages=body.messages,
                model="",  
                temperature=body.temperature,
                max_tokens=body.max_tokens,
                metadata={
                    'enable_l0_retrieval': body.enable_l0_retrieval,
                    'enable_l1_retrieval': body.enable_l1_retrieval,
                    'role_id': body.role_id
                }
            )
            
            # Use ChatService to build enhanced messages (only for knowledge retrieval and prompt building)
            try:
                enhanced_messages = chat_service._build_messages(temp_chat_request)
                logger.info(f"Enhanced messages with local knowledge: {len(enhanced_messages)} messages")
            except Exception as e:
                logger.error(f"Local knowledge retrieval failed: {str(e)}")
                # If knowledge retrieval fails, continue using original messages
                enhanced_messages = body.messages

        # 3. Create CloudService instance
        cloud_service = CloudService()

        try:
            # 4. Call run_inference method using enhanced messages
            response = cloud_service.run_inference(
                messages=enhanced_messages,
                model_id=body.model_id,
                stream=body.stream,
                temperature=body.temperature,
                max_tokens=body.max_tokens
            )
            
            # 5. Handle streaming or non-streaming response
            if body.stream:
                return response
            else:
                # For non-streaming response, return complete JSON response
                if not response:
                    return jsonify(APIResponse.error("Failed to run inference"))
                return jsonify(APIResponse.success(data=response))
                
        except ValueError as e:
            error_msg = str(e)
            logger.error(f"Inference error: {error_msg}")
            error_response = {
                "error": {
                    "message": error_msg,
                    "type": "server_error",
                    "code": "inference_error"
                }
            }
            
            # Return error response based on request type
            if body.stream:
                return local_llm_service.handle_stream_response(iter([error_response]))
            else:
                return jsonify(APIResponse.error(message=error_msg))
                
    except Exception as e:
        logger.error(f"Run inference failed: {str(e)}", exc_info=True)
        return jsonify(APIResponse.error(f"Failed to run inference: {str(e)}"))

@cloud_bp.route("/train/list_deployments", methods=["GET"])
def list_deployments():
    """Get list of all deployed models"""
    try:
        cloud_service = CloudService()
        deployments = cloud_service.list_deployments()
        
        if deployments is None:
            return jsonify(APIResponse.error("Failed to list deployments"))
        
        return jsonify(APIResponse.success(data={
            "deployments": deployments,
            "count": len(deployments)
        }))
    except Exception as e:
        logger.error(f"List deployments failed: {str(e)}", exc_info=True)
        return jsonify(APIResponse.error(f"Failed to list deployments: {str(e)}"))


@cloud_bp.route("/train/delete_deployment", methods=["POST"])
def delete_cloud_deployment():
    """Delete deployed model"""
    try:
        data = request.json
        model_id = data.get("model_id")
        
        if not model_id:
            return jsonify(APIResponse.error("model_id is required"))
        
        cloud_service = CloudService()
        
        status = cloud_service.delete_deployment(model_id)
        
        if status is None:
            return jsonify(APIResponse.error("Failed to delete deployment"))
        
        return jsonify(APIResponse.success(message=f"Deployment deleted successfully", data=status))
    except Exception as e:
        logger.error(f"Delete deployment failed: {str(e)}", exc_info=True)
        return jsonify(APIResponse.error(f"Failed to delete deployment: {str(e)}"))

@cloud_bp.route("/train/delete_fine_tune_job", methods=["POST"])
def delete_fine_tune_job():
    """Delete fine-tune job"""
    try:
        data = request.json
        job_id = data.get("job_id")
        
        if not job_id:
            return jsonify(APIResponse.error("job_id is required"))
        
        cloud_service = CloudService()
        
        success = cloud_service.delete_fine_tune_job(job_id)
        
        if not success:
            return jsonify(APIResponse.error("Failed to delete fine-tune job"))
        
        return jsonify(APIResponse.success(message=f"Fine-tune job deleted successfully"))
    except Exception as e:
        logger.error(f"Delete fine-tune job failed: {str(e)}", exc_info=True)
        return jsonify(APIResponse.error(f"Failed to delete fine-tune job: {str(e)}"))

@cloud_bp.route("/service/start", methods=["POST"])
def start_cloud_service():
    """Start cloud inference service
    
    Request: JSON object, containing:
    - deployment_model: str, Cloud model deployment model
    
    Returns:
    {
        "code": int,
        "message": str,
        "data": {
            "service_type": "cloud",
            "deployment_model": str,
            "status": "active"
        }
    }
    """
    global _cloud_service_active, _cloud_service_model_id
    
    try:
        data = request.get_json()
        if not data or "deployment_model" not in data:
            return jsonify(APIResponse.error(message="Missing required parameter: deployment_model", code=400))

        deployment_model = data["deployment_model"]

        # Check if any service is already running
        current_status = get_service_status()
        if current_status and current_status.get("status") == "active":
            return jsonify(APIResponse.error(
                message=f"Another service is already running: {current_status.get('service_type', 'unknown')}",
                code=400
            ))

        # Verify the cloud model deployment exists
        cloud_service = CloudService()
        try:
            # Verify deployment exists by checking deployed_model field
            deployments = cloud_service.list_deployments()
            model_name = deployment_model

            found = False
            for dep in deployments:
                if dep.get("deployed_model") == deployment_model:
                    model_name = dep.get("name")
                    found = True
                    break
                    
            if not found:
                return jsonify(APIResponse.error(
                    message=f"Cloud model deployment '{deployment_model}' not found",
                    code=404
                ))
        except Exception as e:
            return jsonify(APIResponse.error(message=f"Failed to verify model deployment: {str(e)}"))

        # Create service status file for cloud service
        model_data = {
            "model_id": deployment_model,
            "model_name": model_name,
            "model_path": f"cloud/{deployment_model}",
            "service_endpoint": "cloud_inference"
        }
        
        create_service_status_file("cloud", model_data)

        # Set global status
        _cloud_service_active = True
        _cloud_service_model_id = deployment_model

        logger.info(f"Cloud service started with model: {deployment_model}")
        
        return jsonify(APIResponse.success(
            data={
                "service_type": "cloud",
                "model_id": deployment_model,
                "status": "active"
            },
            message="Cloud inference service started successfully"
        ))

    except Exception as e:
        error_msg = f"Failed to start cloud service: {str(e)}"
        logger.error(error_msg)
        return jsonify(APIResponse.error(message=error_msg, code=500))


@cloud_bp.route("/service/stop", methods=["POST"])
def stop_cloud_service():
    """Stop cloud inference service
    
    Returns:
    {
        "code": int,
        "message": str,
        "data": {
            "service_type": "cloud",
            "status": "stopped"
        }
    }
    """
    global _cloud_service_active, _cloud_service_model_id
    
    try:
        # Check if cloud service is actually running
        current_status = get_service_status()
        if not current_status or current_status.get("service_type") != "cloud":
            return jsonify(APIResponse.error(
                message="No cloud service is currently running",
                code=400
            ))

        # Remove service status file
        remove_service_status_file()

        # Clear global status
        _cloud_service_active = False
        _cloud_service_model_id = None

        logger.info("Cloud service stopped successfully")
        
        return jsonify(APIResponse.success(
            data={
                "service_type": "cloud",
                "status": "stopped"
            },
            message="Cloud inference service stopped successfully"
        ))

    except Exception as e:
        error_msg = f"Failed to stop cloud service: {str(e)}"
        logger.error(error_msg)
        return jsonify(APIResponse.error(message=error_msg, code=500))


@cloud_bp.route("/service/status", methods=["GET"])
def get_cloud_service_status():
    """Get cloud inference service status
    
    Returns:
    {
        "code": int,
        "message": str,
        "data": {
            "service_type": "cloud" | null,
            "model_id": str | null,
            "status": "active" | "stopped",
            "model_data": dict | null
        }
    }
    """
    try:
        current_status = get_service_status()
        
        if current_status and current_status.get("service_type") == "cloud":
            return jsonify(APIResponse.success(
                data={
                    "service_type": "cloud",
                    "model_id": current_status.get("model_data", {}).get("model_id"),
                    "status": "active",
                    "model_data": current_status.get("model_data")
                }
            ))
        else:
            return jsonify(APIResponse.success(
                data={
                    "service_type": None,
                    "model_id": None,
                    "status": "stopped",
                    "model_data": None
                }
            ))

    except Exception as e:
        error_msg = f"Failed to get cloud service status: {str(e)}"
        logger.error(error_msg)
        return jsonify(APIResponse.error(message=error_msg, code=500))

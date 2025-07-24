import time
from werkzeug.utils import secure_filename
from flask import Blueprint, jsonify, Response, request
from charset_normalizer import from_path
from pathlib import Path
import json
from datetime import datetime

from lpm_kernel.api.domains.trainprocess.trainprocess_service import TrainProcessService
from lpm_kernel.api.domains.trainprocess.training_params_manager import TrainingParamsManager
from ...common.responses import APIResponse
from threading import Thread

from lpm_kernel.configs.logging import get_train_process_logger

logger = get_train_process_logger()

trainprocess_bp = Blueprint("trainprocess", __name__, url_prefix="/api/trainprocess")


@trainprocess_bp.route("/start", methods=["POST"])
def start_process():
    """
    Start training process, returns progress stream ID
    
    Request parameters:
        model_name: Model name
        learning_rate: Learning rate for model training (optional)
        number_of_epochs: Number of training epochs (optional)
        concurrency_threads: Number of threads for concurrent processing (optional)
        data_synthesis_mode: Mode for data synthesis (optional)
        use_cuda: Whether to use CUDA for training (optional)
    
    Includes the following steps:
    1. Health check
    2. Generate L0
    3. Generate document embeddings
    4. Process document chunks
    5. Generate chunk embeddings
    6. Analyze documents
    7. Generate stage2
    8. Download model
    9. Prepare data
    10. Train model
    11. Merge weights
    12. Convert model

    Returns:
        Response: JSON response
        {
            "code": 0 for success, non-zero for failure,
            "message": "Error message",
            "data": {
                "progress_id": "Progress stream ID",
                "model_name": "Model name"
            }
        }
    """
    logger.info("Training process starting...")  # Log the startup
    try:
        data = request.get_json()
        if not data or "model_name" not in data:
            return jsonify(APIResponse.error(message="Missing required parameters"))

        model_name = data["model_name"]

        learning_rate = data.get("learning_rate", None)
        number_of_epochs = data.get("number_of_epochs", None)
        concurrency_threads = data.get("concurrency_threads", None)
        data_synthesis_mode = data.get("data_synthesis_mode", None)
        use_cuda = data.get("use_cuda", False)
        is_cot = data.get("is_cot", None)
        language = data.get("language", "en")
        
        # Data filtering parameters
        data_filtering_model = data.get("data_filtering_model", "gemma:2b")
        data_filtering_workers = data.get("data_filtering_workers", 5)
        data_filtering_keep_ratio = data.get("data_filtering_keep_ratio", 0.8)

        # Log the received parameters
        logger.info(
            f"Training parameters: model_name={model_name}, learning_rate={learning_rate}, number_of_epochs={number_of_epochs}, concurrency_threads={concurrency_threads}, data_synthesis_mode={data_synthesis_mode}, is_cot={is_cot}, language={language}, data_filtering_model={data_filtering_model}, data_filtering_workers={data_filtering_workers}, data_filtering_keep_ratio={data_filtering_keep_ratio}")

        # Create service instance with model name and additional parameters
        last_train_service = TrainProcessService.get_instance()

        # Check if there are any in_progress statuses that need to be reset
        if last_train_service is not None and last_train_service.progress.progress.data["status"] == "in_progress":
            return jsonify(APIResponse.error(
                message="There is an existing training process that was interrupted.",
                code=409  # Conflict status code
            ))

        train_service = TrainProcessService(current_model_name=model_name)
        if not train_service.check_training_condition():
            train_service.reset_progress()

        # Save training parameters
        training_params = {
            "model_name": model_name,
            "learning_rate": learning_rate,
            "number_of_epochs": number_of_epochs,
            "concurrency_threads": concurrency_threads,
            "data_synthesis_mode": data_synthesis_mode,
            "use_cuda": use_cuda,  # Make sure to include use_cuda parameter
            "data_filtering_model": data_filtering_model,
            "data_filtering_workers": data_filtering_workers,
            "data_filtering_keep_ratio": data_filtering_keep_ratio,
            "is_cot": is_cot,
            "language": language  # Add language parameter
        }

        params_manager = TrainingParamsManager()
        # Update the latest training parameters
        params_manager.update_training_params(training_params)

        # Log training parameters
        logger.info(f"Saved training parameters: {training_params}")

        thread = Thread(target=train_service.start_process)
        thread.daemon = True
        thread.start()

        # Return success response with all parameters
        return jsonify(
            APIResponse.success(
                data={
                    "model_name": model_name,
                    "learning_rate": learning_rate,
                    "number_of_epochs": number_of_epochs,
                    "concurrency_threads": concurrency_threads,
                    "data_synthesis_mode": data_synthesis_mode,
                    "use_cuda": use_cuda,  # Include in response
                    "is_cot": is_cot,
                    "language": language  # Add language parameter
                }
            )
        )

    except Exception as e:
        logger.error(f"Training process failed: {str(e)}")
        return jsonify(APIResponse.error(message=f"Training process error: {str(e)}"))


@trainprocess_bp.route("/logs", methods=["GET"])
def stream_logs():
    """Get training logs in real-time"""
    log_file_path = "logs/train/train.log"  # Log file path
    last_position = 0

    def generate_logs():
        nonlocal last_position
        while True:
            try:
                encoding = from_path(log_file_path).best().encoding
                with open(log_file_path, 'r', encoding=encoding) as log_file:
                    log_file.seek(last_position)
                    new_lines = log_file.readlines()  # Read new lines

                    for line in new_lines:
                        # Skip empty lines
                        if not line.strip():
                            continue

                        yield f"data: {line.strip()}\n\n"

                    last_position = log_file.tell()
                    if not new_lines:
                        yield f":heartbeat\n\n"
            except Exception as e:
                # If file reading fails, record error and continue
                yield f"data: Error reading log file: {str(e)}\n\n"

            time.sleep(1)  # Check for new logs every second

    return Response(
        generate_logs(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache, no-transform',
            'X-Accel-Buffering': 'no',
            'Connection': 'keep-alive',
            'Transfer-Encoding': 'chunked'
        }
    )


@trainprocess_bp.route("/progress/<model_name>", methods=["GET"])
def get_progress(model_name):
    """Get current progress (non-real-time)"""
    sanitized_model_name = secure_filename(model_name)  # Sanitize model_name
    try:
        train_service = TrainProcessService(current_model_name=sanitized_model_name)  # Pass in specific progress file
        progress = train_service.progress.progress

        return jsonify(
            APIResponse.success(
                data=progress.to_dict()  # Return progress data
            )
        )
    except Exception as e:
        logger.error(f"Get progress failed: {str(e)}", exc_info=True)
        return jsonify(APIResponse.error(message=str(e)))


@trainprocess_bp.route("/progress/reset", methods=["POST"])
def reset_progress():
    """
    Reset progress

    Returns:
        Response: JSON response
        {
            "code": 0 for success, non-zero for failure,
            "message": "Error message",
            "data": null
        }
    """
    try:
        train_service = TrainProcessService.get_instance()
        if train_service is not None:
            train_service.progress.reset_progress()
            logger.info("Progress reset successfully")
        else:
            logger.warning("No active training process found")

        return jsonify(APIResponse.success(message="Progress reset successfully"))
    except Exception as e:
        logger.error(f"Reset progress failed: {str(e)}", exc_info=True)
        return jsonify(APIResponse.error(message=f"Failed to reset progress: {str(e)}"))


@trainprocess_bp.route("/stop", methods=["POST"])
def stop_training():
    """Stop training process
    
    Sets the stop flag and returns immediately. The training process will stop
    after the current step completes. Use the /check_stop_status endpoint to
    monitor the stopping progress.
    """
    try:
        # Get the TrainProcessService instance
        logger.info("Stopping training process...")
        train_service = TrainProcessService.get_instance()
        if train_service is None:
            return jsonify(APIResponse.error(message="Failed to stop training: No active training process"))

        # Stop the process with wait_for_step_completion=True to ensure graceful stopping
        train_service.stop_process()

        # Return immediately
        return jsonify(APIResponse.success(
            data={"status": "success"}
        ))

    except Exception as e:
        logger.error(f"Error stopping training process: {str(e)}", exc_info=True)
        return jsonify(APIResponse.error(message=f"Error stopping training process: {str(e)}"))


@trainprocess_bp.route("/check_stop_status", methods=["GET"])
def check_stop_status():
    """Check if the training process has successfully stopped
    
    Returns:
        JSON response with status information:
        - success: true if the process has been suspended or failed
        - status: the current status of the training process
    """
    try:

        train_service = TrainProcessService.get_instance()
        if train_service is None:
            return jsonify(APIResponse.error(message="Failed to stop training: No active training process"))
        progress = train_service.progress.progress
        logger.info(f"Progress: {progress.data["status"]}")
        if progress.data["status"] == "suspended" or progress.data["status"] == "failed":
            return jsonify(APIResponse.success(
                message="Training process has been stopped and status is confirmed as suspended",
                data={"status": "success"}
            ))
        else:
            return jsonify(APIResponse.success(
                message="Training process has been stopped and status is confirmed as failed",
                data={"status": "pending"}
            ))


    except Exception as e:
        logger.error(f"Error checking stop status: {str(e)}", exc_info=True)
        return jsonify(APIResponse.error(message=f"Error checking stop status: {str(e)}"))


@trainprocess_bp.route("/step_output_content", methods=["GET"])
def get_step_output_content():
    """
    Get content of output file for a specific training step
    
    Request parameters:
        step_name: Name of the step to get content for, e.g. 'extract_dimensional_topics'
    
    Returns:
        Response: JSON response
        {
            "code": 0,
            "message": "Success",
            "data": {...}  // Content of the output file, or null if not found
        }
    """
    try:
        # Get TrainProcessService instance
        train_service = TrainProcessService.get_instance()
        if train_service is None:
            logger.error("No active training process found.")
            return jsonify(APIResponse.error(message="No active training process found."))

        # Get step name from query parameters
        step_name = request.args.get('step_name')
        if not step_name:
            return jsonify(APIResponse.error(message="Missing required parameter: step_name", code=400))

        # Get step output content
        output_content = train_service.get_step_output_content(step_name)
        logger.info(f"Step output content: {output_content}")
        return jsonify(APIResponse.success(data=output_content))
    except Exception as e:
        logger.error(f"Failed to get step output content: {str(e)}", exc_info=True)
        return jsonify(APIResponse.error(message=f"Failed to get step output content: {str(e)}"))


@trainprocess_bp.route("/training_params", methods=["GET"])
def get_training_params():
    """
    Get the latest training parameters for both local and cloud training
    
    Returns:
        Response: JSON response
        {
            "code": 0 for success, non-zero for failure,
            "message": "Error message",
            "data": {
                "local": {
                    "model_name": "Model name",
                    "learning_rate": "Learning rate",
                    "number_of_epochs": "Number of epochs",
                    "concurrency_threads": "Concurrency threads",
                    "data_synthesis_mode": "Data synthesis mode"
                },
                "cloud": {
                    "model_name": "Model name",
                    "base_model": "Base model name",
                    "training_type": "Training type",
                    "hyper_parameters": {
                        "n_epochs": "Number of epochs",
                        "learning_rate": "Learning rate"
                    },
                    "created_at": "Creation timestamp"
                }
            }
        }
    """
    try:
        # Get the latest local training parameters
        params_manager = TrainingParamsManager()
        local_training_params = params_manager.get_latest_training_params()

        # Get the latest cloud training parameters
        project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
        cloud_params_file = project_root / "data/cloud_progress/cloud_training_params.json"

        if cloud_params_file.exists():
            try:
                with open(cloud_params_file, 'r', encoding='utf-8') as f:
                    cloud_training_params = json.load(f)
            except Exception as e:
                logger.error(f"Failed to load cloud training parameters: {str(e)}", exc_info=True)
                cloud_training_params = get_default_cloud_params()
        else:
            logger.warning(f"Cloud training parameters file does not exist: {cloud_params_file}, using default values")
            cloud_training_params = get_default_cloud_params()

        # Combine both parameters
        combined_params = {
            "local": local_training_params,
            "cloud": cloud_training_params
        }

        return jsonify(APIResponse.success(data=combined_params))
    except Exception as e:
        logger.error(f"Error getting training parameters: {str(e)}", exc_info=True)
        return jsonify(APIResponse.error(message=f"Error getting training parameters: {str(e)}"))


def get_default_cloud_params():
    """Return default cloud training parameters"""
    current_time = datetime.now()
    timestamp = current_time.strftime("%Y%m%d_%H%M%S")
    return {
        "model_name": timestamp,
        "base_model": "qwen2.5-7b-instruct",
        "training_type": "efficient_sft",
        "hyper_parameters": {
            "n_epochs": 1
        },
        "data_synthesis_mode": "low",
        "language": "en",
        "created_at": current_time.isoformat()
    }


@trainprocess_bp.route("/retrain", methods=["POST"])
def retrain():
    """
    Reset progress to data processing stage (data_processing not started) and automatically start the training process
    
    Request parameters:
        model_name: Model name (required)
        learning_rate: Learning rate for model training (optional)
        number_of_epochs: Number of training epochs (optional)
        concurrency_threads: Number of threads for concurrent processing (optional)
        data_synthesis_mode: Mode for data synthesis (optional)
        use_cuda: Whether to use CUDA for training (optional)
        is_cot: Whether to use Chain of Thought (optional)
    
    Returns:
        Response: JSON response
        {
            "code": 0 for success, non-zero for failure,
            "message": "Error message",
            "data": {
                "progress_id": "Progress stream ID",
                "model_name": "Model name"
            }
        }
    """
    try:
        # get request parameters
        data = request.get_json() or {}
        model_name = data.get("model_name")

        if not model_name:
            return jsonify(APIResponse.error(message="missing necessary parameter: model_name", code=400))

        # Get optional parameters
        learning_rate = data.get("learning_rate", None)
        number_of_epochs = data.get("number_of_epochs", None)
        concurrency_threads = data.get("concurrency_threads", None)
        data_synthesis_mode = data.get("data_synthesis_mode", None)
        use_cuda = data.get("use_cuda", False)
        is_cot = data.get("is_cot", None)
        language = data.get("language", "en")  # Add language parameter, default to English

        # Create training service instance
        train_service = TrainProcessService(current_model_name=model_name)

        # Check if there are any in_progress statuses that need to be reset
        if train_service.progress.progress.data["status"] == "in_progress":
            # Reset the progress and continue
            logger.info("There is an existing training process that was interrupted.")

        train_service.reset_progress()

        # Save training parameters
        training_params = {
            "model_name": model_name,
            "learning_rate": learning_rate,
            "number_of_epochs": number_of_epochs,
            "concurrency_threads": concurrency_threads,
            "data_synthesis_mode": data_synthesis_mode,
            "use_cuda": use_cuda,
            "is_cot": is_cot,
            "language": language  # Add language parameter
        }

        params_manager = TrainingParamsManager()
        # Update the training parameters, optionally using previous params as base
        params_manager.update_training_params(training_params, use_previous_params=False)

        # Log training parameters
        logger.info(f"Saved training parameters: {training_params}")

        thread = Thread(target=train_service.start_process)
        thread.daemon = True
        thread.start()

        return jsonify(
            APIResponse.success(
                message="Successfully reset progress to data processing stage and started training process",
                data={
                    "model_name": model_name,
                    "learning_rate": learning_rate,
                    "number_of_epochs": number_of_epochs,
                    "concurrency_threads": concurrency_threads,
                    "data_synthesis_mode": data_synthesis_mode,
                    "use_cuda": use_cuda,
                    "is_cot": is_cot,
                    "language": language  # Add language parameter
                }
            )
        )
    except Exception as e:
        logger.error(f"Retrain reset failed: {str(e)}", exc_info=True)
        return jsonify(APIResponse.error(message=f"Failed to reset progress to data processing stage: {str(e)}"))

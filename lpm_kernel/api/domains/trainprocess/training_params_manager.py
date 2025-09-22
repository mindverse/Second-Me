"""
Training parameters management module.
This module provides functions for managing and accessing training parameters.
"""

import logging
import json
import os

# Configure logger
logger = logging.getLogger(__name__)


class TrainingParamsManager:
    """
    Training parameters manager class.
    """
    
    # Default training parameters
    _default_training_params = {
        "model_name": "Qwen2.5-0.5B-Instruct",
        "learning_rate": 1e-4,
        "number_of_epochs": 3,
        "concurrency_threads": 2,
        "data_synthesis_mode": "high",
        "use_cuda": False,  # Default to using CUDA when available
        "is_cot": False
    }
    
    # Parameters file path
    _params_file_path = None
    
    @classmethod
    def _get_params_file_path(cls):
        """
        Get the training parameters file path
        """
        if cls._params_file_path is None:
            # Set the parameters file path
            progress_dir = os.path.join(os.getcwd(), "data", "progress")
            if not os.path.exists(progress_dir):
                os.makedirs(progress_dir)
            cls._params_file_path = os.path.join(progress_dir, "training_params.json")
        
        return cls._params_file_path
    
    @classmethod
    def update_training_params(cls, params, use_previous_params=True):
        """
        Update the latest training parameters and save to file
        
        Args:
            params: Dictionary containing training parameters
            use_previous_params: Whether to use previous training parameters as base
        """
        # First try to load existing parameters
        current_params = cls.get_latest_training_params() if use_previous_params else cls._default_training_params.copy()
        
        # Update parameters
        for key, value in params.items():
            if key in cls._default_training_params:
                current_params[key] = value
                logger.debug(f"Updated training parameter {key} to {value}")
            else:
                logger.warning(f"Ignoring unknown parameter: {key}")
        
        # Save to file
        params_file = cls._get_params_file_path()
        try:
            with open(params_file, 'w', encoding='utf-8') as f:
                json.dump(current_params, f, indent=2)
            logger.info(f"Training parameters saved to {params_file}")
        except Exception as e:
            logger.error(f"Failed to save training parameters to file: {str(e)}", exc_info=True)
    
    @classmethod
    def get_latest_training_params(cls):
        """
        Get the latest training parameters from file
        
        Returns:
            dict: Dictionary containing the latest training parameters
        """
        params_file = cls._get_params_file_path()
        
        # If file exists, read from file
        if os.path.exists(params_file):
            try:
                with open(params_file, 'r', encoding='utf-8') as f:
                    params = json.load(f)
                
                # Replace null values with default values
                default_params = cls._default_training_params.copy()
                for key, value in default_params.items():
                    if key not in params or params[key] is None:
                        params[key] = value
                
                logger.debug(f"Loaded training parameters from {params_file}")
                return params
            except Exception as e:
                logger.error(f"Failed to load training parameters from file: {str(e)}", exc_info=True)
                # If reading fails, return default parameters
                return cls._default_training_params.copy()
        else:
            # If file does not exist, return default parameters
            return cls._default_training_params.copy()
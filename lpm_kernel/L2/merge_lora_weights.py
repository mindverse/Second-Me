"""Utility for merging LoRA weights into base language models.

This module provides functions to merge trained LoRA adapter weights with a base model,
producing a standalone model that incorporates the adaptations without needing the
LoRA architecture during inference.
"""

import argparse
import os
import gc
import sys
import logging
import traceback
import torch
import datetime
from typing import Optional, Dict, Any

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from lpm_kernel.L2.memory_manager import get_memory_manager

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def merge_lora_weights(base_model_path, lora_adapter_path, output_model_path):
    """Merge LoRA weights into a base model and save the result.
    
    This function loads a base model and a LoRA adapter, merges them together,
    and saves the resulting model to the specified output path. It leverages
    PyTorch's built-in memory management features.
    
    Args:
        base_model_path: Path to the base model directory.
        lora_adapter_path: Path to the LoRA adapter directory.
        output_model_path: Path where the merged model will be saved.
    """
    # Get memory manager
    memory_manager = get_memory_manager()
    
    try:
        # Log initial memory state
        memory_info = memory_manager.get_memory_info()
        logger.info(f"Initial memory state: RAM used: {memory_info['ram_used_gb']:.2f}GB, "
                   f"available: {memory_info['ram_available_gb']:.2f}GB")
        
        # Determine if CUDA is available and should be used
        use_cuda = memory_manager.cuda_available
        device = "cuda" if use_cuda else "cpu"
        
        if use_cuda:
            logger.info(f"CUDA is available. VRAM used: {memory_info.get('vram_used_gb', 0):.2f}GB")
        else:
            logger.warning("CUDA not available or not enabled. Using CPU for model operations.")
        
        # Clean up memory before starting
        memory_manager.cleanup_memory(force=True)
        
        # Use auto dtype selection instead of manually choosing based on hardware
        device_map = "auto" if use_cuda else None
        
        logger.info(f"Loading base model from {base_model_path} with device_map={device_map}, using auto dtype")
        
        # Use auto dtype configuration for optimal hardware utilization
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype="auto",
            device_map=device_map
        )
        
        # Load tokenizer - this doesn't consume much memory
        tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        
        # Load the LoRA adapter and apply it to the base model
        logger.info(f"Loading LoRA adapter from {lora_adapter_path}")
        lora_model = PeftModel.from_pretrained(base_model, lora_adapter_path)
        
        # Merge weights - this is done automatically by PyTorch on appropriate devices
        logger.info(f"Merging LoRA weights into base model on {device}")
        merged_model = lora_model.merge_and_unload()
        
        # Clean up before saving
        memory_manager.cleanup_memory()
        
        # Add inference optimization config to the merged model for faster startup
        if use_cuda:
            # Set inference-specific configuration in model config
            if hasattr(merged_model.config, "torch_dtype"):
                merged_model.config.torch_dtype = "float16"  # Prefer float16 for inference
            if not hasattr(merged_model.config, "pretraining_tp"):
                merged_model.config.pretraining_tp = 1  # For tensor parallelism during inference
            
            # Set default inference device
            if not hasattr(merged_model.config, "_default_inference_device"):
                merged_model.config._default_inference_device = "cuda:0"
            
            logger.info("Added GPU optimization settings to model configuration")
        
        # Save merged model with shard size to prevent OOM errors during save
        logger.info(f"Saving merged model to {output_model_path}")
        merged_model.save_pretrained(
            output_model_path,
            safe_serialization=True,
            max_shard_size="2GB"  # Sharded saving to avoid memory spikes
        )
        tokenizer.save_pretrained(output_model_path)
        
        # Save a special marker file to indicate this model should use GPU for inference
        if use_cuda:
            with open(os.path.join(output_model_path, "gpu_optimized.json"), "w") as f:
                import json
                json.dump({"gpu_optimized": True, "optimized_on": datetime.datetime.now().isoformat()}, f)
                logger.info("Added GPU optimization marker file for faster service startup")
        
        logger.info("Model successfully merged and saved!")
        
    except Exception as e:
        logger.error(f"Error during model merge: {str(e)}")
        logger.error(traceback.format_exc())
        # Force cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise


def merge_model_weights(
    base_model_path="resources/L2/base_models",
    lora_adapter_path="resources/model/output/personal_model",
    output_model_path="resources/model/output/merged_model",
):
    """Merge LoRA weights into base model with default paths.
    
    This is a convenience function that calls merge_lora_weights with default 
    paths that match the expected directory structure of the project.

    Args:
        base_model_path: Path to the base model. Defaults to "resources/L2/base_models".
        lora_adapter_path: Path to the LoRA adapter. Defaults to "resources/model/output/personal_model".
        output_model_path: Path to save the merged model. Defaults to "resources/model/output/merged_model".
    """
    merge_lora_weights(base_model_path, lora_adapter_path, output_model_path)


def parse_arguments():
    """Parse command line arguments for the script.
    
    Returns:
        argparse.Namespace: The parsed command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Merge LoRA weights into a base model."
    )
    parser.add_argument(
        "--base_model_path", type=str, required=True, help="Path to the base model."
    )
    parser.add_argument(
        "--lora_adapter_path", type=str, required=True, help="Path to the LoRA adapter."
    )
    parser.add_argument(
        "--output_model_path",
        type=str,
        required=True,
        help="Path to save the merged model.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    merge_lora_weights(
        args.base_model_path, args.lora_adapter_path, args.output_model_path
    )

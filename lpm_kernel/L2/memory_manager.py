"""Memory management utilities for PyTorch training.

This module provides lightweight utilities to monitor memory usage
and configure PyTorch's built-in memory management features.
"""

import os
import gc
import logging
import psutil
import torch
from typing import Dict, Any

# Configure logging
logger = logging.getLogger(__name__)

class MemoryManager:
    """Simple memory manager that leverages PyTorch's built-in memory optimizations."""
    
    def __init__(self):
        """Initialize the memory manager."""
        self.cuda_available = torch.cuda.is_available()
        self.process = psutil.Process(os.getpid())
        
        # Remove redundant environment variable setting - now handled in train_for_user.sh
    
    def get_memory_info(self) -> Dict[str, Any]:
        """Get current memory usage information."""
        info = {
            "ram_used_percent": psutil.virtual_memory().percent,
            "ram_used_gb": psutil.virtual_memory().used / (1024**3),
            "ram_available_gb": psutil.virtual_memory().available / (1024**3),
            "ram_total_gb": psutil.virtual_memory().total / (1024**3),
        }
        
        if self.cuda_available:
            try:
                info.update({
                    "vram_used_gb": torch.cuda.memory_allocated() / (1024**3),
                    "vram_reserved_gb": torch.cuda.memory_reserved() / (1024**3),
                    "vram_total_gb": torch.cuda.get_device_properties(0).total_memory / (1024**3),
                })
            except RuntimeError as e:
                logger.warning(f"Error getting CUDA memory info: {str(e)}")
                self.cuda_available = False
        
        return info
    
    def cleanup_memory(self, force: bool = False) -> None:
        """Free up memory by garbage collection and emptying CUDA cache."""
        # Run Python garbage collection
        gc.collect()
        
        # Empty CUDA cache if available
        if self.cuda_available:
            torch.cuda.empty_cache()
            
        # Log memory status after cleanup
        if force:
            info = self.get_memory_info()
            logger.info(
                f"Memory after cleanup: RAM: {info['ram_used_gb']:.2f}GB / {info['ram_total_gb']:.2f}GB, "
                f"VRAM: {info.get('vram_used_gb', 0):.2f}GB / {info.get('vram_total_gb', 0):.2f}GB"
            )
    
    def get_optimal_training_config(self) -> Dict[str, Any]:
        """Get recommended configurations for model training based on hardware capabilities."""
        # Default configs that rely on PyTorch's automatic memory management
        config = {
            "device_map": "auto",
            "fp16": False,
            "bf16": False,
            "gradient_checkpointing": True,
            "gradient_accumulation_steps": 1,
        }
        
        # Let PyTorch automatically decide the best dtype if CUDA is available
        if self.cuda_available:
            # Instead of manually checking capabilities, use "auto" dtype
            # PyTorch will automatically select the best precision for the hardware
            config["dtype"] = "auto"
            
            # Adjust accumulation steps based on available memory
            vram_gb = self.get_memory_info().get("vram_total_gb", 0)
            if vram_gb < 8:  # Small GPUs
                config["gradient_accumulation_steps"] = 4
            elif vram_gb < 16:  # Medium GPUs
                config["gradient_accumulation_steps"] = 2
        
        return config
    
    def optimize_model_for_training(self, model):
        """Apply PyTorch's built-in memory optimizations for training."""
        # Enable gradient checkpointing if available
        if hasattr(model, "gradient_checkpointing_enable"):
            logger.info("Enabling gradient checkpointing for memory efficiency")
            model.gradient_checkpointing_enable()
            
        # Enable memory-efficient attention for PyTorch 2.0+
        if hasattr(model, "config"):
            try:
                model.config.use_memory_efficient_attention = True
            except:
                pass
            
            # Enable flash attention for compatible GPUs
            if self.cuda_available and torch.cuda.get_device_capability()[0] >= 8:
                try:
                    model.config.attn_implementation = "flash_attention_2"
                except:
                    pass
        
        return model
    
    def optimize_training_args(self, training_args):
        """Configure training arguments for efficient memory usage."""
        if not training_args:
            return None
            
        # Get optimal configuration based on hardware
        config = self.get_optimal_training_config()
        
        # Apply configurations to training arguments
        if not getattr(training_args, "fp16", False) and not getattr(training_args, "bf16", False):
            training_args.fp16 = config["fp16"]
            training_args.bf16 = config["bf16"]
        
        if not getattr(training_args, "gradient_checkpointing", False):
            training_args.gradient_checkpointing = config["gradient_checkpointing"]
        
        if training_args.gradient_accumulation_steps == 1:
            training_args.gradient_accumulation_steps = config["gradient_accumulation_steps"]
        
        logger.info("Training configuration optimized for memory efficiency:")
        logger.info(f"  Mixed precision: FP16={training_args.fp16}, BF16={training_args.bf16}")
        logger.info(f"  Gradient checkpointing: {training_args.gradient_checkpointing}")
        logger.info(f"  Gradient accumulation steps: {training_args.gradient_accumulation_steps}")
        
        return training_args


# Global memory manager instance
memory_manager = MemoryManager()

def get_memory_manager() -> MemoryManager:
    """Get the global memory manager instance."""
    return memory_manager
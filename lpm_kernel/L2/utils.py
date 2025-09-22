"""Utility functions for the L2 model training and inference.

This module provides utilities for token counting, model preparation, data processing,
and other helper functions used across the L2 pipeline.
"""

from collections import defaultdict
from datetime import datetime
from enum import Enum
import json
import os
import sys

from datasets import DatasetDict, Dataset, load_dataset, load_from_disk
from datasets.builder import DatasetGenerationError
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
import tiktoken
import torch
import logging
from lpm_kernel.configs.logging import TRAIN_LOG_FILE

from lpm_kernel.L2.training_prompt import (
    CONTEXT_PROMPT,
    CONTEXT_COT_PROMPT,
    JUDGE_PROMPT,
    JUDGE_COT_PROMPT,
    MEMORY_PROMPT,
    MEMORY_COT_PROMPT,
)

# Add import for memory manager
from .memory_manager import get_memory_manager
import gc
import requests

# Initialize the logger
logger = logging.getLogger(__name__)

# Default chat templates for different model formats
DEFAULT_CHATML_CHAT_TEMPLATE = "{% for message in messages %}\n{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% if loop.last and add_generation_prompt %}{{'<|im_start|>assistant\n' }}{% endif %}{% endfor %}"
DEFAULT_ZEPHYR_CHAT_TEMPLATE = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"


def release_ollama_models_early():
    """Release Ollama models from memory as early as possible before model loading.
    
    This function uses the Ollama API with keep_alive=0 parameter to properly unload models
    and free up VRAM before loading the training model.
    """
    try:
        from lpm_kernel.api.services.user_llm_config_service import UserLLMConfigService
        import json
        
        logger.info("Early release of Ollama models to free up VRAM for training")
        
        # Get current user LLM config to identify models to release
        user_llm_config_service = UserLLMConfigService()
        user_llm_config = user_llm_config_service.get_available_llm()
        
        if not user_llm_config:
            logger.warning("No user LLM configuration found. Skipping Ollama model release.")
            return
        
        # Track which models have been released
        released_models = set()
        
        def get_generate_url(base_endpoint):
            """Helper function to get the API endpoint for unloading models"""
            if not base_endpoint:
                return None
                
            base_url = base_endpoint.rstrip("/")
            
            # Convert to API base URL if needed (may be v1 format or direct ollama format)
            if "/v1/" in base_url:
                api_base = base_url.split("/v1/")[0]
                return f"{api_base}/api/generate"
            else:
                # Check if this is a non-localhost Ollama instance
                if "ollama" in base_url.lower():
                    if "localhost" in base_url or "127.0.0.1" in base_url:
                        return "http://localhost:11434/api/generate"
                    else:
                        # Extract the base URL and use it
                        parts = base_url.split("//")
                        if len(parts) > 1:
                            host = parts[1].split("/")[0]
                            return f"{parts[0]}//{host}/api/generate"
                    
                # Default ollama endpoint as fallback
                return "http://localhost:11434/api/generate"
        
        # Release chat model if using Ollama
        if "ollama" in user_llm_config.chat_endpoint.lower() and user_llm_config.chat_model_name:
            chat_model = user_llm_config.chat_model_name
            generate_url = get_generate_url(user_llm_config.chat_endpoint)
            
            if not generate_url:
                logger.warning(f"Could not determine API endpoint for chat model: {chat_model}")
            else:
                logger.info(f"Releasing Ollama chat model: {chat_model} via {generate_url}")
                
                try:
                    # Set up headers with API key if provided
                    headers = {
                        "Content-Type": "application/json"
                    }
                    if user_llm_config.chat_api_key:
                        headers["Authorization"] = f"Bearer {user_llm_config.chat_api_key}"
                    
                    # Use the proper generate endpoint with keep_alive=0 to unload
                    payload = {
                        "model": chat_model,
                        "keep_alive": 0,
                        "prompt": " "  # Minimal prompt needed for request
                    }
                    
                    unload_response = requests.post(
                        generate_url,
                        headers=headers,
                        data=json.dumps(payload),
                        timeout=30  # Add timeout to prevent hanging
                    )
                    
                    if unload_response.status_code < 300:
                        logger.info(f"✅ Successfully unloaded chat model: {chat_model}")
                        released_models.add(chat_model)
                    else:
                        logger.warning(f"Failed to unload model via API: {unload_response.status_code} - {unload_response.text}")
                except Exception as e:
                    logger.warning(f"Failed to release chat model {chat_model}: {str(e)}")
        
        # Release embedding model if different from chat model and using Ollama
        if (user_llm_config.embedding_model_name and 
            "ollama" in user_llm_config.embedding_endpoint.lower() and
            user_llm_config.embedding_model_name != user_llm_config.chat_model_name and
            user_llm_config.embedding_model_name not in released_models):
            
            embedding_model = user_llm_config.embedding_model_name
            generate_url = get_generate_url(user_llm_config.embedding_endpoint)
            
            if not generate_url:
                logger.warning(f"Could not determine API endpoint for embedding model: {embedding_model}")
            else:
                logger.info(f"Releasing Ollama embedding model: {embedding_model} via {generate_url}")
                
                try:
                    # Set up headers with API key if provided
                    headers = {
                        "Content-Type": "application/json"
                    }
                    if user_llm_config.embedding_api_key:
                        headers["Authorization"] = f"Bearer {user_llm_config.embedding_api_key}"
                    
                    # Use the proper generate endpoint with keep_alive=0 to unload
                    payload = {
                        "model": embedding_model,
                        "keep_alive": 0,
                        "prompt": " "  # Minimal prompt needed for request
                    }
                    
                    unload_response = requests.post(
                        generate_url,
                        headers=headers,
                        data=json.dumps(payload),
                        timeout=30  # Add timeout to prevent hanging
                    )
                    
                    if unload_response.status_code < 300:
                        logger.info(f"✅ Successfully unloaded embedding model: {embedding_model}")
                        released_models.add(embedding_model)
                    else:
                        logger.warning(f"Failed to unload model via API: {unload_response.status_code} - {unload_response.text}")
                except Exception as e:
                    logger.warning(f"Failed to release embedding model {embedding_model}: {str(e)}")
        
        # Final cleanup and verification
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            memory_info = get_memory_manager().get_memory_info()
            vram_used = memory_info.get('vram_used_gb', 0)
            vram_total = memory_info.get('vram_total_gb', 1)
            logger.info(f"VRAM after early model release: {vram_used:.2f}GB / {vram_total:.2f}GB ({vram_used/vram_total*100:.1f}%)")
        
        if released_models:
            logger.info(f"Early release completed for {len(released_models)} Ollama models: {', '.join(released_models)}")
        else:
            logger.info("No Ollama models were released early")
    
    except Exception as e:
        import traceback
        logger.error(f"Error in early Ollama model release: {str(e)}")
        logger.error(traceback.format_exc())


def count_tokens_from_string(string: str, encoding_name: str = "cl100k_base") -> int:
    """Returns the number of tokens in a text string using a specified encoding.

    Args:
        string: Text to tokenize.
        encoding_name: The encoding name to use. Defaults to "cl100k_base".

    Returns:
        The number of tokens in the text.
    """
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def truncate_string_by_tokens(
    string: str, max_tokens: int, encoding_name: str = "cl100k_base"
) -> str:
    """Truncates a string to fit within a specified number of tokens.
    
    Args:
        string: Text to truncate.
        max_tokens: Maximum number of tokens to keep.
        encoding_name: The encoding name to use. Defaults to "cl100k_base".
        
    Returns:
        The truncated string.
    """
    encoding = tiktoken.get_encoding(encoding_name)
    tokens = encoding.encode(string)
    if len(tokens) > max_tokens:
        # Truncate the tokens to the maximum token limit
        truncated_tokens = tokens[:max_tokens]
        # Decode the truncated tokens back to a string
        truncated_string = encoding.decode(truncated_tokens)
        return truncated_string
    return string

class ChatmlSpecialTokens(str, Enum):
    """Special tokens for ChatML format models."""
    user = "<|im_start|>user"
    assistant = "<|im_start|>assistant"
    system = "<|im_start|>system"
    eos_token = "<|im_end|>"
    bos_token = "<s>"
    pad_token = "<pad>"

    @classmethod
    def list(cls):
        """Returns a list of all special tokens."""
        return [token.value for token in cls]

class ZephyrSpecialTokens(str, Enum):
    """Special tokens for Zephyr format models."""
    user = "<|user|>"
    assistant = "<|assistant|>"
    system = "<|system|>"
    eos_token = "</s>"
    bos_token = "<s>"
    pad_token = "<pad>"

    @classmethod
    def list(cls):
        """Returns a list of all special tokens."""
        return [token.value for token in cls]

def create_and_prepare_model(args, data_args, training_args, model_kwargs=None):
    """Creates and prepares a model for training.
    
    Args:
        args: Model arguments containing model configuration.
        data_args: Data arguments for training.
        training_args: Training configuration arguments.
        model_kwargs: Additional kwargs to pass to the model loading function.
        
    Returns:
        Tuple of (model, tokenizer, peft_config) ready for training.
    """
    # Get the memory manager for adaptive loading
    memory_manager = get_memory_manager()
    model_kwargs = model_kwargs or {}
    
    # Release Ollama models early before we load any models
    if torch.cuda.is_available() and args.use_cuda:
        release_ollama_models_early()
        # Force cleanup memory after releasing Ollama models
        memory_manager.cleanup_memory(force=True)
    
    if args.use_unsloth:
        from unsloth import FastLanguageModel
    bnb_config = None

    if (
        torch.distributed.is_available()
        and torch.distributed.is_initialized()
        and torch.distributed.get_world_size() > 1
        and args.use_unsloth
    ):
        raise NotImplementedError("Unsloth is not supported in distributed training")

    # Clean up memory before loading model
    memory_manager.cleanup_memory()
    
    # Check for CUDA availability and use it if enabled
    cuda_available = torch.cuda.is_available()
    use_cuda_requested = args.use_cuda
    device = "cpu"

    # Always enable memory-adaptive loading by default (device_map="auto"), unless CUDA is off
    if cuda_available and use_cuda_requested:
        device = "cuda"
        model_kwargs["device_map"] = "auto"
    else:
        if use_cuda_requested and not cuda_available:
            logger.warning("⚠️ CUDA was requested but is not available on this system. Falling back to CPU.")
        elif cuda_available and not use_cuda_requested:
            logger.info("ℹ️ CUDA is available but not requested. Using CPU as specified.")
        else:
            logger.info("ℹ️ CUDA is not available. Using CPU for training.")
        # Explicitly remove device_map to force CPU-only
        if "device_map" in model_kwargs:
            model_kwargs.pop("device_map")
        logger.info("Using CPU for model training and inference.")

    # Configure quantization based on available memory
    # Use model_kwargs quantization_config if provided, otherwise build it
    if "quantization_config" not in model_kwargs:
        if args.use_4bit_quantization:
            compute_dtype = getattr(torch, args.bnb_4bit_compute_dtype)
            quant_storage_dtype = getattr(torch, args.bnb_4bit_quant_storage_dtype)

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=args.use_4bit_quantization,
                bnb_4bit_quant_type=args.bnb_4bit_quant_type,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=args.use_nested_quant,
                bnb_4bit_quant_storage=quant_storage_dtype,
            )
            model_kwargs["quantization_config"] = bnb_config

            if compute_dtype == torch.float16 and args.use_4bit_quantization:
                major, _ = torch.cuda.get_device_capability() if torch.cuda.is_available() else (0, 0)
                if major >= 8:
                    logger.info("Your GPU supports bfloat16, you can accelerate training with the argument --bf16")
        elif args.use_8bit_quantization:
            bnb_config = BitsAndBytesConfig(load_in_8bit=args.use_8bit_quantization)
            model_kwargs["quantization_config"] = bnb_config

    # Load model with memory-adaptive approach
    model = None
    tokenizer = None
    peft_config = None
    
    try:
        # First try loading the model with the requested configuration
        if args.use_unsloth:
            # Load model with Unsloth using memory manager
            unsloth_kwargs = {
                "model_name": args.model_name_or_path,
                "max_seq_length": data_args.max_seq_length,
                "dtype": None,
                "load_in_4bit": args.use_4bit_quantization,
                "load_in_8bit": args.use_8bit_quantization,
                "trust_remote_code": True,
                "device_map": model_kwargs.get("device_map", "auto") if args.use_cuda and torch.cuda.is_available() else None,
            }
            
            logger.info(f"Loading model with Unsloth with parameters: {unsloth_kwargs}")
            model, _ = FastLanguageModel.from_pretrained(**unsloth_kwargs)
            
        else:
            # Load model with standard approach
            load_kwargs = {
                "trust_remote_code": True,
            }
            
            # Use any provided model_kwargs
            load_kwargs.update(model_kwargs)
            
            if "attn_implementation" not in load_kwargs and args.use_flash_attn:
                load_kwargs["attn_implementation"] = "flash_attention_2"
            
            # Set default device_map if not specified
            if "device_map" not in load_kwargs and args.use_cuda and torch.cuda.is_available():
                load_kwargs["device_map"] = "auto"
                            
            logger.info(f"Loading model with parameters: {load_kwargs}")
            model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **load_kwargs)
    
    except (RuntimeError, torch.cuda.OutOfMemoryError, MemoryError) as e:
        # If standard approaches fail, try progressive fallbacks
        logger.warning(f"Failed to load model with standard settings: {str(e)}")
        logger.info("Falling back to adaptive model loading...")
        
        # First cleanup to ensure maximum memory available
        memory_manager.cleanup_memory(force=True)
        
        try:
            # Try with simpler configuration - float16 instead of bfloat16
            logger.info("Attempting to load with float16 precision...")
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name_or_path,
                device_map="auto" if torch.cuda.is_available() and args.use_cuda else None,
                torch_dtype=torch.float16 if torch.cuda.is_available() and args.use_cuda else None,
                trust_remote_code=True
            )
        except (RuntimeError, torch.cuda.OutOfMemoryError, MemoryError) as e:
            # If that fails too, try even more conservative loading
            logger.warning(f"Float16 loading failed: {str(e)}")
            memory_manager.cleanup_memory(force=True)
            
            try:
                # Try with CPU offloading and gradual loading
                logger.info("Attempting most conservative loading with CPU offloading...")
                model = AutoModelForCausalLM.from_pretrained(
                    args.model_name_or_path,
                    device_map="auto",
                    offload_folder="offload_folder",
                    offload_state_dict=True,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else None,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                )
            except Exception as e:
                # If all fallbacks fail, it's a fatal error
                logger.error(f"All adaptive loading approaches failed: {str(e)}")
                raise RuntimeError(f"Failed to load model with any memory adaptation technique: {str(e)}")

    # If still not loaded, it's a fatal error
    if model is None:
        raise RuntimeError("Failed to load model with any memory adaptation technique")

    # Apply memory optimization to model
    model = memory_manager.optimize_model_for_training(model)

    # Configure LoRA if requested
    if args.use_peft_lora and not args.use_unsloth:
        peft_config = LoraConfig(
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            r=args.lora_r,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=args.lora_target_modules.split(",")
            if args.lora_target_modules != "all-linear"
            else args.lora_target_modules,
        )
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, trust_remote_code=True, padding_side="right"
    )

    # Apply Unsloth LoRA if requested and check memory status
    if args.use_unsloth:
        try:
            # Clean up first
            memory_manager.cleanup_memory()
            
            # Apply LoRA with memory monitoring
            model = FastLanguageModel.get_peft_model(
                model,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                r=args.lora_r,
                target_modules=args.lora_target_modules.split(",")
                if args.lora_target_modules != "all-linear"
                else args.lora_target_modules,
                use_gradient_checkpointing=training_args.gradient_checkpointing,
                random_state=training_args.seed,
                max_seq_length=data_args.max_seq_length,
            )
            
        except Exception as e:
            logger.error(f"Failed to apply Unsloth LoRA: {str(e)}")
            # If Unsloth fails, we might need to fall back to standard training
            if args.use_cuda and torch.cuda.is_available():
                logger.warning("Low VRAM detected, moving model to CPU")
                model = model.cpu()
                torch.cuda.empty_cache()

    # Final memory status check
    memory_info = memory_manager.get_memory_info()
    logger.info(f"Memory after model preparation: RAM: {memory_info['ram_used_gb']:.2f}GB / {memory_info['ram_total_gb']:.2f}GB")
    if torch.cuda.is_available():
        logger.info(f"VRAM: {memory_info.get('vram_used_gb', 0):.2f}GB / {memory_info.get('vram_total_gb', 0):.2f}GB")

    return model, peft_config, tokenizer


def create_chat_data(data_args, tokenizer):
    """Creates and preprocesses chat data for training.
    
    Args:
        data_args: Arguments for dataset configuration.
        tokenizer: Tokenizer for text processing.
        
    Returns:
        Processed dataset ready for training.
    """
    def preprocess(sample, user_name='user', is_cot=False):
        """Preprocesses a chat sample.
        
        Args:
            sample: The input sample to process.
            user_name: Name of the user. Defaults to 'user'.
            is_cot: Whether to use chain-of-thought prompts. Defaults to False.
            
        Returns:
            Processed chat sample.
        """
        if sample.get('assistant') is None and sample.get('enhanced_request') is not None:
            user_message = f"{user_name}'s request is: " + sample['user_request']
            messages = [
                {"role": "system", "content": CONTEXT_COT_PROMPT.format(user_name=user_name) if is_cot else CONTEXT_PROMPT.format(user_name=user_name)},
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": sample['enhanced_request'].strip('\n')},
            ]
            return [{"content": tokenizer.apply_chat_template(messages, tokenize=False)}]
        if sample.get('assistant') is None and sample.get('user_feedback') is not None:
            user_message = f"{user_name}'s request is: " + sample['user_request'] + "\n" + "Expert's response is: " + sample['expert_response']
            messages = [
                {"role": "system", "content": JUDGE_COT_PROMPT.format(user_name=user_name) if is_cot else JUDGE_PROMPT.format(user_name=user_name)},
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": sample['user_feedback'].strip('\n')},
            ]
            return [{"content": tokenizer.apply_chat_template(messages, tokenize=False)}]
        
        if sample.get('assistant') is None:
            return []
        sample['assistant'] = sample['assistant'].strip('\n')
        
        messages = [
            {"role": "system", "content": MEMORY_COT_PROMPT.format(user_name=user_name) if is_cot else MEMORY_PROMPT.format(user_name=user_name)},
            {"role": "user", "content": sample['user']},
            {"role": "assistant", "content": sample['assistant']},
        ]
        if 'None' in sample['assistant']:
            return []
        return [{"content": tokenizer.apply_chat_template(messages, tokenize=False)}]
    
    dataset = load_dataset("json", data_files=data_args.dataset_name, split="train")
    res_dataset = []
    
    for case in dataset:
        res_dataset.extend(preprocess(case, data_args.user_name, data_args.is_cot))
    
    res = Dataset.from_list(res_dataset)
    print(f"**************Dataset contains {res.num_rows} elements.**************")

    return res


def formatting_prompts_func(example):
    """Format examples for training.
    
    Args:
        example: Example to format.
        
    Returns:
        Formatted text.
    """
    out_text_list = []
    for i in range(len(example["content"])):
        out_text_list.append(example["content"][i])
    return out_text_list

# Improved logging setup
def setup_logger(log_path, logger_name="download_logger"):
    """Setup a logger with file and console handlers."""
    # Create logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    
    # Remove any existing handlers to avoid duplicates
    if logger.handlers:
        for handler in logger.handlers:
            logger.removeHandler(handler)
    
    # Create file handler
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def save_hf_model(model_name=None, log_file_path=None) -> str:
    """Saves a Hugging Face model locally.
    
    Args:
        model_name: Name of the model to save. If None, will attempt to get from config.
        log_file_path: Path to save download logs. If None, uses default path.
        
    Returns:
        Path to the saved model.
    """
    # If log_file_path is None or empty, use default path
    if not log_file_path:
        log_file_path = TRAIN_LOG_FILE
    
    # Setup logging
    logger = setup_logger(log_file_path)
    
    # If no model name provided, attempt to get from training configuration
    if not model_name:
        try:
            from lpm_kernel.configs.config import Config
            config = Config()
            model_name = config.get("training", {}).get("model_name")
            if not model_name:
                logger.warning("No model name provided and none found in config. Using Qwen2.5-0.5B-Instruct as fallback.")
                model_name = "Qwen2.5-0.5B-Instruct"
        except Exception as e:
            logger.warning(f"Failed to get model name from config: {str(e)}. Using Qwen2.5-0.5B-Instruct as fallback.")
            model_name = "Qwen2.5-0.5B-Instruct"
    
    base_dir = os.path.join(os.getcwd(), "resources/L2/base_models")
    # Normalize model name and check for path traversal attempts
    normalized_model_name = os.path.normpath(model_name)
    if ".." in normalized_model_name or normalized_model_name.startswith("/"):
        raise ValueError("Invalid model name - potential path traversal attempt")
    
    # Prepare save path
    save_path = os.path.join(base_dir, normalized_model_name)
    os.makedirs(save_path, exist_ok=True)

    from huggingface_hub import list_repo_files, configure_http_backend, hf_hub_download
    from tqdm import tqdm
    from concurrent.futures import ThreadPoolExecutor
    import traceback
    
    # Configure HTTP backend more simply
    try:
        configure_http_backend(timeout=100.0)
    except Exception as e:
        logger.warning(f"Failed to configure HTTP backend with timeout: {e}")
        try:
            configure_http_backend()
        except Exception as e:
            logger.warning(f"Failed to configure HTTP backend: {e}")

    # Log download start
    logger.info(f"Starting download of model: {model_name}")
    logger.info(f"Will be saved to: {save_path}")
    
    hf_model_name = f"Qwen/{model_name}"
    
    try:
        # Get list of files to download
        files = list_repo_files(hf_model_name)
        logger.info(f"Found {len(files)} files to download from {hf_model_name}")

        def download_file_with_progress(filename):
            """Download a single file from the model repository"""
            local_file_path = os.path.join(save_path, filename)
            
            # Create directories if they don't exist
            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
            
            # Check if file already exists and is not empty
            if os.path.exists(local_file_path) and os.path.getsize(local_file_path) > 0:
                logger.info(f"File already exists: {filename}")
                return True
            
            try:
                # Build the download URL
                hf_endpoint = os.environ.get("HF_ENDPOINT")
                url = f"{hf_endpoint}/{hf_model_name}/resolve/main/{filename}"
                
                # Get file size
                response = requests.head(url)
                total_size = int(response.headers.get('content-length', 0))
                
                # If the size cannot be obtained, set a default value
                if total_size == 0:
                    logger.info(f"Starting download of file: {filename} (Size unknown)")
                else:
                    logger.info(f"Starting download of file: {filename} (Size: {total_size / 1024 / 1024:.2f} MB)")
                
                # Create the file to write to
                with open(local_file_path, 'wb') as f:
                    # Create a progress bar
                    progress_bar = tqdm(
                        total=total_size if total_size > 0 else None,
                        unit='iB',
                        unit_scale=True,
                        desc=f"Downloading {os.path.basename(filename)}",
                        disable=False
                    )
                    
                    # Define progress callback
                    def progress_callback(current, total):
                        progress_bar.update(current - progress_bar.n)
                        
                        # Log progress every ~10MB
                        if current % (1024 * 1024 * 10) < 8192:
                            if total and total > 0:
                                percent = current / total * 100
                                logger.info(f"File {filename}: Downloaded {current/1024/1024:.2f} MB / {total/1024/1024:.2f} MB ({percent:.2f}%)")
                            else:
                                logger.info(f"File {filename}: Downloaded {current/1024/1024:.2f} MB (total size unknown)")
                
                    # Download file with progress tracking
                    response = requests.get(url, stream=True)
                    if response.status_code == 200:
                        downloaded = 0
                        
                        # Update total size if needed
                        actual_total = int(response.headers.get('content-length', 0))
                        if actual_total > 0 and (total_size == 0 or total_size != actual_total):
                            total_size = actual_total
                            logger.info(f"Updated file size for {filename}: {total_size / 1024 / 1024:.2f} MB")
                            progress_bar.total = total_size
                            progress_bar.refresh()
                        
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                                downloaded += len(chunk)
                                progress_callback(downloaded, total_size)
                                
                        progress_bar.close()
                        logger.info(f"Completed download of file: {filename}")
                        return True
                    else:
                        logger.error(f"Failed to download {filename}: HTTP status {response.status_code}")
                        failed_files.append(filename)
                        return False
                        
            except Exception as e:
                logger.error(f"Failed to download {filename}: {str(e)}")
                failed_files.append(filename)
                return False

        # Keep track of failed files for potential retry
        failed_files = []
        
        # Use ThreadPoolExecutor for parallel downloads with controlled concurrency
        max_workers = min(8, len(files))  # Limit concurrent downloads to avoid overloading
        successful_downloads = 0
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Create a progress bar for overall download progress
            with tqdm(total=len(files), desc="Downloading model files") as progress:
                futures = [executor.submit(download_file_with_progress, file) for file in files]
                
                for future in futures:
                    result = future.result()
                    if result:
                        successful_downloads += 1
                    progress.update(1)
                    
                    # Report progress periodically
                    if progress.n % 5 == 0 or progress.n == len(files):
                        logger.info(f"Downloaded {progress.n}/{len(files)} files ({successful_downloads} successful)")
        
        # Handle any failed downloads
        if failed_files:
            logger.warning(f"Failed to download {len(failed_files)} files. First few: {failed_files[:5]}")
            
            # If most files failed, there might be an issue with the model repository
            if len(failed_files) > len(files) * 0.5:
                logger.error(f"More than 50% of files failed to download. There might be an issue with the model repository.")
                raise RuntimeError("Too many files failed to download")
            
            # If critical files failed (like model weights or config), warn specifically
            critical_patterns = ['model.safetensors', 'config.json', 'tokenizer.json']
            critical_failed = [f for f in failed_files if any(pattern in f for pattern in critical_patterns)]
            if critical_failed:
                logger.error(f"Failed to download critical files: {critical_failed}")
                raise RuntimeError(f"Failed to download critical model files: {', '.join(critical_failed)}")
        
        # Record the download completion information
        try:
            import glob
            file_count = len(glob.glob(f"{save_path}/**/*", recursive=True))
            logger.info(f"Model {model_name} downloaded with {file_count} files.")
        except Exception:
            logger.info(f"Download completed for model: {model_name}.")
    except requests.RequestException:
        try:
            from modelscope.hub.snapshot_download import snapshot_download
            snapshot_download(model_id=hf_model_name, local_dir=save_path)
        except Exception as e:
            logger.error(f"Error downloading model: {str(e)}")
            raise
    except KeyboardInterrupt:
        logger.warning(f"Download interrupted by user for model: {model_name}")
        # Clean up partial downloads
        raise
    except Exception as e:
        logger.error(f"Error downloading model: {str(e)}")
        logger.error(traceback.format_exc())
        raise
    return save_path

def format_timestr(utc_time_str):
    """Formats a UTC time string to a more readable format.
    
    Args:
        utc_time_str: UTC time string to format.
        
    Returns:
        Formatted time string.
    """
    # Define the original time format
    try:
        # Parse the UTC time
        utc_time = datetime.strptime(utc_time_str, "%Y-%m-%d %H:%M:%S%z")
        
        # Convert to readable format
        formatted_time = utc_time.strftime("%B %d, %Y at %I:%M %p")
        
        return formatted_time
    except ValueError:
        # Handle invalid date format
        return utc_time_str


if __name__ == "__main__":
    if len(sys.argv) > 1:
        save_hf_model(sys.argv[1])
    else:
        save_hf_model()

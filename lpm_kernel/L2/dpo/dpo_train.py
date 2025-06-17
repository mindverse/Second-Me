import json
import argparse
import torch

try:
    from unsloth import FastLanguageModel, FastTokenizer
    UNSLOTH_AVAILABLE = True
except ImportError:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    UNSLOTH_AVAILABLE = False

from datasets import Dataset
from trl import DPOConfig, DPOTrainer
from peft import LoraConfig
from datetime import datetime, timedelta
# from clearml import Task

def get_east_eight_time_formatted():
    # Get the current UTC time
    utc_now = datetime.utcnow()
    # Convert UTC time to East Eight Time by adding 8 hours
    east_eight_time = utc_now + timedelta(hours=8)
    # Format the time as mmdd-hhmm
    formatted_time = east_eight_time.strftime("%m%d-%H%M")
    return formatted_time

# task = Task.init(project_name="mind_dpo", task_name="qwen25-instruct-" + get_east_eight_time_formatted())

def get_supported_dtype():
    # Try bf16, fallback to f16
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16, "bfloat16"
        else:
            return torch.float16, "float16"
    try:
        _ = torch.zeros(1, dtype=torch.bfloat16)
        return torch.bfloat16, "bfloat16"
    except Exception:
        return torch.float16, "float16"

def training_data_processor(args, SYS = "You are a helpful assistant.\n\n"):
    with open(args.training_data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    training_data = {
        "prompt": [
            [
                {"role": "system", "content": data_point['prompt']['system']},
                {"role": "user", "content": data_point['prompt']['user']}
            ] for data_point in data
        ], 
        "chosen": [data_point["chosen"] for data_point in data],
        "rejected": [data_point["rejected"] for data_point in data]
    }
    if UNSLOTH_AVAILABLE:
        tokenizer = FastTokenizer.from_pretrained(args.base_model_path, padding_side="left")
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, padding_side="left")
    training_data = {
        "prompt": tokenizer.apply_chat_template(training_data["prompt"], tokenize=False),
        "chosen": training_data["chosen"],
        "rejected": training_data["rejected"]
    }
    return training_data

def train(args):
    dtype, dtype_str = get_supported_dtype()
    if UNSLOTH_AVAILABLE:
        tokenizer = FastTokenizer.from_pretrained(args.base_model_path, padding_side="left")
        model = FastLanguageModel.from_pretrained(
            model_name=args.base_model_path,
            dtype=dtype_str,
            load_in_4bit=False,
            load_in_8bit=False,
            device_map="auto" if torch.cuda.is_available() else "cpu"
        )
        # Apply LoRA with Unsloth's optimized method if requested
        if args.lora_r > 0:
            model = FastLanguageModel.get_peft_model(
                model,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                r=args.lora_r,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                use_gradient_checkpointing=False,
                random_state=42,
                max_seq_length=args.max_length,
            )
        # Use FastDPOTrainer if available, else fallback
        try:
            from unsloth import FastDPOTrainer as DPOTrainerImpl
        except ImportError:
            from trl import DPOTrainer as DPOTrainerImpl
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, padding_side="left")
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model_path, 
            trust_remote_code=True,
            ignore_mismatched_sizes=True, 
            torch_dtype=dtype,
        )
        DPOTrainerImpl = DPOTrainer
    time_str = get_east_eight_time_formatted()

    data_dict = training_data_processor(args)
    dataset = Dataset.from_dict(data_dict)

    # Only use LoRA config for non-Unsloth
    lora_config = None
    if not UNSLOTH_AVAILABLE and args.lora_r > 0:
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            task_type="CAUSAL_LM",
            inference_mode=False,
            fan_in_fan_out=False
        )

    training_args = DPOConfig(
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        gradient_checkpointing=False,
        gradient_checkpointing_kwargs={"use_reentrant":True},
        max_grad_norm=args.max_grad_norm,
        lr_scheduler_type="cosine",
        logging_steps=5,
        optim="adamw_hf",  # use the optimizer suits cpu
        loss_type="sigmoid",
        warmup_steps=args.warmup_steps,
        warmup_ratio=args.warmup_ratio,
        do_eval=False,
        max_prompt_length=1024,
        max_length=args.max_length,
        seed=42,
        output_dir="resources/model/output/dpo_model/adapter",
        remove_unused_columns=False,
        fp16=False, 
        bf16=False,
        beta=args.beta,
    )

    dpo_trainer = DPOTrainerImpl(
        model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset,
        peft_config=lora_config,
    )

    dpo_trainer.train()
    dpo_trainer.save_model()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DPO Training Stage')

    # Model arguments
    parser.add_argument('--training_data_path', type=str, required=False, 
                       default="resources/L2/data/dpo/dpo_direct.json")
    parser.add_argument('--base_model_path', type=str, required=False,
                       default="resources/model/output/merged_model")
    
    # Training arguments
    parser.add_argument('--num_train_epochs', type=int, default=5)
    parser.add_argument('--learning_rate', type=float, default=5e-6)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--max_grad_norm', type=float, default=0.3)
    parser.add_argument('--warmup_steps', type=int, default=10)
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument('--max_length', type=int, default=2048)
    parser.add_argument('--beta', type=float, default=0.1)
    
    # LoRA arguments
    parser.add_argument('--lora_r', type=int, default=16)
    parser.add_argument('--lora_alpha', type=int, default=16)
    parser.add_argument('--lora_dropout', type=float, default=0.0)

    args = parser.parse_args()
    
    # task = Task.init(project_name="mind_dpo", 
    #                 task_name=f"qwen25-instruct-{get_east_eight_time_formatted()}")
    
    train(args)

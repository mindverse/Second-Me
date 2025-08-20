import json
import argparse
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset
from trl import DPOConfig, DPOTrainer
from peft import LoraConfig, AutoPeftModelForCausalLM, get_peft_model
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
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, padding_side="left")
    training_data = {
        "prompt": tokenizer.apply_chat_template(training_data["prompt"], tokenize=False),
        "chosen": training_data["chosen"],
        "rejected": training_data["rejected"]
    }
    return training_data

def train(args):
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path, 
        trust_remote_code=True,
        ignore_mismatched_sizes=True, 
        torch_dtype="auto",  # Use auto detection instead of hardcoding float32
    )
    time_str = get_east_eight_time_formatted()

    # merged_model = model.merge_and_unload()
    # merged_model.save_pretrained(merged_model)

    data_dict = training_data_processor(args)
    dataset = Dataset.from_dict(data_dict)

    if args.lora_r == 0:
        lora_config = None
    else:
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            target_modules="all-linear",
            task_type="CAUSAL_LM",
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

    dpo_trainer = DPOTrainer(
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
    parser.add_argument('--lora_r', type=int, default=64)
    parser.add_argument('--lora_alpha', type=int, default=128)
    parser.add_argument('--lora_dropout', type=float, default=0.1)

    # DeepSpeed arguments
    parser.add_argument('--deepspeed', type=str, default=None)
    parser.add_argument('--local_rank', type=int, default=-1)

    args = parser.parse_args()
    
    # task = Task.init(project_name="mind_dpo", 
    #                 task_name=f"qwen25-instruct-{get_east_eight_time_formatted()}")
    
    train(args)

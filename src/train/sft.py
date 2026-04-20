import json
import random
import dotenv
import numpy as np
from tqdm import tqdm
dotenv.load_dotenv()

import sys
import os
os.environ["WANDB_MODE"]    = "offline"
os.environ["WANDB_DISABLED"]= "true"
import warnings
from dataclasses import asdict, dataclass, field
from typing import Optional, List, Dict, Any

warnings.filterwarnings("ignore", category=FutureWarning)
import logging

import torch.distributed as dist

def is_main_process():
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    return True

if is_main_process():
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True
    )
else:
    logger = logging.getLogger(__name__)
    logger.addHandler(logging.NullHandler())

import trl
from datasets import load_dataset
import transformers
import torch
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset

@dataclass
class TrainingConfig:
    model_name: str = field(default="Qwen/Qwen2.5-7B-Instruct")
    block_size: int = field(default=4000)
    train_file_path: Optional[str] = field(default="")
    use_flash_attention_2: bool = field(default=False)
    
    eval_split_ratio: float = field(default=0.01) 
    
    use_peft: bool = field(default=False)  
    
    lora_r: int = field(default=32)
    lora_alpha: int = field(default=16)
    lora_dropout: float = field(default=0.05)

    use_data_collator: bool = field(default=False)

def train():
    parser = transformers.HfArgumentParser((TrainingConfig, trl.SFTConfig))
    config, args = parser.parse_args_into_dataclasses()

    transformers.set_seed(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    log_config = {**asdict(config), **asdict(args)}
    logger.info(f"Initial training config: {log_config}")
    
    # loading model
    kwargs = {}
    if config.use_flash_attention_2:
        logger.info(f"Use flash_attention_2")
        kwargs["attn_implementation"] = "flash_attention_2"
    
    model = transformers.AutoModelForCausalLM.from_pretrained(config.model_name, **kwargs)
    
    if config.use_peft:
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=None,  
            bias="none",
        )
        logger.info(f"Using PEFT-LoRA: {peft_config}")
        model = get_peft_model(model, peft_config)
        
        if args.gradient_checkpointing:
            model.enable_input_require_grads()
        
        model.print_trainable_parameters()

    if config.train_file_path and os.path.isfile(config.train_file_path):
        logger.info(f"Loading QA data from local JSONL: {config.train_file_path}")
        records = []
        with open(config.train_file_path, "r", encoding="utf-8") as _f:
            for _line in _f:
                _line = _line.strip()
                if not _line:
                    continue
                records.append(json.loads(_line))
        raw = Dataset.from_list(records)
        dataset = {"train": raw}
    else:
        logger.info(f"Loading data from HuggingFace: {config.train_file_path}")
        dataset = load_dataset(config.train_file_path)
    
    if "subq_rewrite" in config.train_file_path:
        def preprocess_subq_rewrite(x):
            subq      = x["subquestion"].strip()
            grounded  = x.get("grounded_text", "").strip()
            rewritten = x["rewritten_text"].strip()
            context   = x.get("context", "").strip()

            if context:
                user_content = f"Context:\n{context}\n\nQuestion:\n{subq}"
            else:
                user_content = subq

            output_content = rewritten

            return {
                "prompt": [{"role": "user", "content": user_content}],
                "completion": [
                    {"role": "assistant", "content": output_content}
                ],
            }
        dataset["train"] = dataset["train"].map(preprocess_subq_rewrite, remove_columns=dataset["train"].column_names)
    elif "chain_rewrite" in config.train_file_path:
        def preprocess_chain_rewrite(x):
            prompt   = x.get("prompt", "").strip()
            think    = x.get("think_content_rewritten", "").strip()
            partial   = x.get("partial_response_before", "").strip()

            user_content = prompt 

            final_resp = f"<think>\n{think}\n</think>\n\n{partial}"
            return {
                "prompt": [{"role": "user", "content": user_content}],
                "completion": [
                    {"role": "assistant", "content": final_resp}
                ],
            }
        dataset["train"] = dataset["train"].map(preprocess_chain_rewrite, remove_columns=dataset["train"].column_names)


    # setting up trainer
    tokenizer = transformers.AutoTokenizer.from_pretrained(config.model_name, use_fast=True)
    tokenizer.padding_side = 'left'

    

    if config.use_data_collator:
        if "Llama" in config.model_name:
            instruction_template = "<|start_header_id|>user<|end_header_id|>"
            response_template = "<|start_header_id|>assistant<|end_header_id|>\n\n"
            tokenizer.pad_token = "<|reserved_special_token_5|>"
        elif "Qwen" in config.model_name:
            instruction_template = "<|im_start|>user"
            response_template = "<|im_start|>assistant\n"
            tokenizer.pad_token = "<|im_end|>"

        collator = trl.DataCollatorForCompletionOnlyLM(
            instruction_template=instruction_template,
            response_template=response_template,
            tokenizer=tokenizer,
            mlm=False
        )
        args.dataset_text_field="text"
    else:
        collator = None
    
    args.max_length = config.block_size

    if not hasattr(args, "report_to") or not args.report_to:
        args.report_to = ["swanlab"]
    else:
        if isinstance(args.report_to, str):
            args.report_to = [args.report_to]
        
        if "swanlab" not in args.report_to:
            args.report_to = args.report_to + ["swanlab"]

    if args.do_eval and "test" not in dataset:
        train_size = int((1 - config.eval_split_ratio) * len(dataset["train"]))
        eval_size = len(dataset["train"]) - train_size
        
        split_dataset = dataset["train"].train_test_split(
            test_size=eval_size, 
            seed=args.seed,
        )
        
        dataset = {
            "train": split_dataset["train"],
            "test": split_dataset["test"]
        }
        
        logger.info(f"Split dataset: Training set {len(dataset['train'])} samples, Evaluation set {len(dataset['test'])} samples")
    elif not args.do_eval:
        logger.info("Evaluation disabled, will use all data for training")

    if config.use_data_collator:
        if "prompt" in dataset["train"].column_names:
            dataset["train"] = dataset["train"].remove_columns("prompt")
        if "test" in dataset and "prompt" in dataset["test"].column_names:
            dataset["test"] = dataset["test"].remove_columns("prompt")
    

    log_config = {**asdict(config), **asdict(args)}
    logger.info(f"Used training config: {log_config}")
    
    trainer = trl.SFTTrainer(
        model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"] if "test" in dataset and args.do_eval else None,
        args=args,
        data_collator=collator,
    )

    trainer.train()

    if args.output_dir:
        if config.use_peft:
            
            merged_model = trainer.model.merge_and_unload()
            
            merged_model.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)
            
            
            lora_adapter_path = os.path.join(args.output_dir, "lora_adapter")
            trainer.model.save_pretrained(lora_adapter_path)
            
        else:
            trainer.save_model(output_dir=args.output_dir)
            tokenizer.save_pretrained(args.output_dir)

    trainer.accelerator.wait_for_everyone()


if __name__ == "__main__":
    train()

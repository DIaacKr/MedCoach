import logging

from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

logger = logging.getLogger(__name__)

import random
import dotenv
import numpy as np
import torch
dotenv.load_dotenv()

import sys
import os
os.environ["WANDB_MODE"]    = "offline"
os.environ["WANDB_DISABLED"]= "true"
import warnings
from dataclasses import asdict, dataclass, field
from typing import Optional

warnings.filterwarnings("ignore", category=FutureWarning)

import trl
from datasets import load_dataset
import transformers
from transformers import TrainerCallback, TrainerState, TrainerControl
from peft import LoraConfig, TaskType

from callbacks import get_callbacks

class CustomEarlyStoppingCallback(TrainerCallback):
    def __init__(self, patience: int = 3, threshold: float = 0.0):
        self.patience = patience
        self.threshold = threshold
        self.best = None
        self.bad_epochs = 0

    def on_evaluate(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        metrics = kwargs.get("metrics", {})
        current = metrics.get("eval_loss")
        if current is None:
            return control

        if self.best is None or current < self.best - self.threshold:
            self.best = current
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
            if self.bad_epochs >= self.patience:
                control.should_training_stop = True
                control.should_save = True
        return control

@dataclass
class DPOConfig:
    model_name: str = field(default="path/to/your/sft_model")
    
    train_file_path: Optional[str] = field(default="path/to/your/preference_dataset")

    eval_split_ratio: float = field(default=0.01)  
    
    use_peft: bool = field(default=False)  
    use_flash_attention_2: bool = field(default=False)
    
    lora_r: int = field(default=32)
    lora_alpha: int = field(default=64)
    lora_dropout: float = field(default=0.05)

    early_stopping_patience: int = field(
        default=3,
    )
    early_stopping_threshold: float = field(
        default=0.0,
    )

def train():
    parser = transformers.HfArgumentParser((DPOConfig, trl.DPOConfig))
    config, args = parser.parse_args_into_dataclasses()

    transformers.set_seed(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    log_config = {**asdict(config), **asdict(args)}
    logger.info(f"DPO Training config: {log_config}")
    
    # loading model
    kwargs = {}
    if config.use_flash_attention_2:
        logger.info(f"Use flash_attention_2")
        kwargs["attn_implementation"] = "flash_attention_2"

    kwargs["device_map"] = None
    
    model = transformers.AutoModelForCausalLM.from_pretrained(config.model_name, **kwargs)

    if args.gradient_checkpointing:
        logger.info("Enabling gradient checkpointing for DPO...")
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
        
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
    
    peft_config = None
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
        ref_model = None
    else:
        ref_model = transformers.AutoModelForCausalLM.from_pretrained(
            config.model_name,
            **kwargs,
        )
        #ref_model = None 

    if os.path.isfile(config.train_file_path) and config.train_file_path.lower().endswith((".json", ".jsonl")):
        dataset = load_dataset(
            "json",
            data_files={"train": config.train_file_path},
            field=None  
        )
    else:
        dataset = load_dataset(config.train_file_path)
    logger.info(f"Loaded dataset: {config.train_file_path}, Columns: {dataset['train'].column_names}")

    required_columns = ['prompt', 'chosen', 'rejected']
    if not all(col in dataset['train'].column_names for col in required_columns):
        raise ValueError(f"Dataset must contain the following columns: {required_columns}")

    # setting up tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(config.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token 

    if args.do_eval and "test" not in dataset:
        split_dataset = dataset["train"].train_test_split(
            test_size=config.eval_split_ratio, 
            seed=args.seed
        )
        dataset = {
            "train": split_dataset["train"],
            "test": split_dataset["test"]
        }
        logger.info(f"Split dataset: Train set {len(dataset['train'])} samples, Eval set {len(dataset['test'])} samples")
    elif not args.do_eval:
        logger.info("Evaluation is disabled, will use all data for training")

    callbacks = get_callbacks()

    callbacks.append(
        CustomEarlyStoppingCallback(
            patience = config.early_stopping_patience,
            threshold= config.early_stopping_threshold
        )
    )
    
    trainer = trl.DPOTrainer(
        model,
        ref_model=ref_model,
        args=args,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("test") if args.do_eval else None,
        processing_class=tokenizer,
        peft_config=peft_config,
        callbacks=callbacks,
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

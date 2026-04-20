#!/bin/bash

model_name="path/to/your/sft_model"
train_file_path="path/to/your/preference_dataset"
config_file="exps/configs/deepspeed_zero3.yaml" 
output_base="outputs/dpo"
exp_name="dpo"
uid="$(date +%Y%m%d_%H%M%S)"

use_peft=false
lora_r=64
lora_alpha=128
lora_dropout=0.05

use_flash_attention_2=true
gradient_checkpointing=true
beta=0.1 
lr=5e-7 
epochs=3
global_batch_size=16
per_device_batch_size=1
gpu_count=1
eval_split_ratio=0.01

offload_optimizer=false 
offload_param=false
weight_decay=0.1
warmup_ratio=0.1
lr_scheduler_type="cosine"
adam_beta1=0.9
adam_beta2=0.95
max_length=4000 

do_eval=true
eval_strategy="steps"
eval_steps=0.1
save_strategy="no"
save_steps=0.5
logging_steps=10
bf16=true
seed=42

while [[ $# -gt 0 ]]; do
  case $1 in
    --model_name) model_name="$2"; shift 2 ;;
    --train_file_path) train_file_path="$2"; shift 2 ;;
    --config_file) config_file="$2"; shift 2 ;;
    --output_base) output_base="$2"; shift 2 ;;
    --exp_name) exp_name="$2"; shift 2 ;;
    --use_peft) use_peft="$2"; shift 2 ;;
    --use_flash_attention_2) use_flash_attention_2="$2"; shift 2 ;;
    --gradient_checkpointing) gradient_checkpointing="$2"; shift 2 ;;
    --lora_r) lora_r="$2"; shift 2 ;;
    --lora_alpha) lora_alpha="$2"; shift 2 ;;
    --lora_dropout) lora_dropout="$2"; shift 2 ;;
    --eval_split_ratio) eval_split_ratio="$2"; shift 2 ;;
    --beta) beta="$2"; shift 2 ;;
    --lr) lr="$2"; shift 2 ;;
    --epochs) epochs="$2"; shift 2 ;;
    --global_batch_size) global_batch_size="$2"; shift 2 ;;
    --per_device_batch_size) per_device_batch_size="$2"; shift 2 ;;
    --weight_decay) weight_decay="$2"; shift 2 ;;
    --warmup_ratio) warmup_ratio="$2"; shift 2 ;;
    --lr_scheduler_type) lr_scheduler_type="$2"; shift 2 ;;
    --adam_beta1) adam_beta1="$2"; shift 2 ;;
    --adam_beta2) adam_beta2="$2"; shift 2 ;;
    --do_eval) do_eval="$2"; shift 2 ;;
    --eval_strategy) eval_strategy="$2"; shift 2 ;;
    --eval_steps) eval_steps="$2"; shift 2 ;;
    --save_strategy) save_strategy="$2"; shift 2 ;;
    --save_steps) save_steps="$2"; shift 2 ;;
    --logging_steps) logging_steps="$2"; shift 2 ;;
    --bf16) bf16="$2"; shift 2 ;;
    --seed) seed="$2"; shift 2 ;;
    --gpu_count) gpu_count="$2"; shift 2 ;;
    --max_length) max_length="$2"; shift 2 ;;
    --offload_optimizer) offload_optimizer="$2"; shift 2 ;;
    --offload_param) offload_param="$2"; shift 2 ;;
  esac
done

run_config_file=$config_file

if [ "$use_peft" = true ]; then
    echo "=== LoRA Mode Detected ==="
    
    if [[ "$run_config_file" == *"zero3"* ]]; then
        echo "Warning: ZeRO-3 config detected with LoRA. Switching to ZeRO-2 for stability."
        run_config_file="exps/configs/deepspeed_zero2.yaml" 
    fi

fi
if [ "$offload_optimizer" = true ] || [ "$offload_param" = true ]; then
    echo "Processing CPU Offload..."
    temp_config_path=$(dirname "$config_file")/temp_$(basename "$config_file")
    cp "$config_file" "$temp_config_path"
    if [ "$offload_optimizer" = true ]; then
        sed -i 's/offload_optimizer_device: none/offload_optimizer_device: cpu/' "$temp_config_path"
    fi
    if [ "$offload_param" = true ]; then
        sed -i 's/offload_param_device: none/offload_param_device: cpu/' "$temp_config_path"
    fi
    run_config_file=$temp_config_path
fi

grad_acc=$(( global_batch_size / (gpu_count * per_device_batch_size) ))

export NCCL_DEBUG=INFO
export NCCL_TIMEOUT=3600
export MASTER_ADDR=localhost
export MASTER_PORT=$((10000 + RANDOM % 55000))
export WORLD_SIZE=$gpu_count
export RANK=0


short_model_name=$(basename "$model_name")
if [ "$use_peft" = true ]; then
    run_name="${exp_name}_${short_model_name}_lora_r${lora_r}_lr${lr}_epochs${epochs}_${uid}"
else
    run_name="${exp_name}_${short_model_name}_full_lr${lr}_${uid}"
fi
output_dir="${output_base}/${run_name}"
mkdir -p logs "$output_dir"

echo "Run name: $run_name"
echo "Config: $run_config_file"
echo "Gradient Checkpointing: $gradient_checkpointing"

accelerate launch \
  --config_file "$run_config_file" \
  --num_processes $gpu_count \
  --main_process_port $MASTER_PORT \
  --num_machines 1 \
  --machine_rank 0 \
  src/train/dpo.py \
    --report_to swanlab \
    --run_name "$run_name" \
    --model_name "$model_name" \
    --train_file_path "$train_file_path" \
    --eval_split_ratio $eval_split_ratio \
    --use_peft $use_peft \
    --use_flash_attention_2 $use_flash_attention_2 \
    --lora_r $lora_r \
    --lora_alpha $lora_alpha \
    --lora_dropout $lora_dropout \
    --gradient_checkpointing $gradient_checkpointing \
    --beta $beta \
    --learning_rate $lr \
    --per_device_train_batch_size $per_device_batch_size \
    --per_device_eval_batch_size $per_device_batch_size \
    --gradient_accumulation_steps $grad_acc \
    --num_train_epochs $epochs \
    --weight_decay $weight_decay \
    --warmup_ratio $warmup_ratio \
    --lr_scheduler_type $lr_scheduler_type \
    --adam_beta1 $adam_beta1 \
    --adam_beta2 $adam_beta2 \
    --do_eval $do_eval \
    --eval_strategy $eval_strategy \
    --eval_steps $eval_steps \
    --save_strategy $save_strategy \
    --save_steps $save_steps \
    --logging_steps $logging_steps \
    --bf16 $bf16 \
    --seed $seed \
    --max_length ${max_length} \
    --output_dir "$output_dir"

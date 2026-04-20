#!/bin/bash

tp=1
dp=1
max_new_tokens=8300 
batch_size=1024
temperature=0.6
limit=-1
overwrite=True
model_path="path/to/your/model"
tokenizer_path="null"  # 
exp_name=""
output_dir="outputs/eval"
prefix_prompt=""
suffix_prompt="Return your final response within \\boxed{{}}."
timeout=1800
seed=42
mem_fraction_static=0.9
port=$((29000 + RANDOM % 1000))  # 
eval_data_path="data/m1_eval_data.json"
config_path="exps/configs/base.yaml"


while [[ $# -gt 0 ]]; do
    case $1 in
        --help)
            show_help
            exit 0
            ;;
        --tp) tp="$2"; shift 2 ;;
        --dp) dp="$2"; shift 2 ;;
        --max_new_tokens) max_new_tokens="$2"; shift 2 ;;
        --batch_size) batch_size="$2"; shift 2 ;;
        --temperature) temperature="$2"; shift 2 ;;
        --limit) limit="$2"; shift 2 ;;
        --overwrite) overwrite="$2"; shift 2 ;;
        --model_path) model_path="$2"; shift 2 ;;
        --tokenizer_path) tokenizer_path="$2"; shift 2 ;;
        --exp_name) exp_name="$2"; shift 2 ;;
        --output_dir) output_dir="$2"; shift 2 ;;
        --prefix_prompt) prefix_prompt="$2"; shift 2 ;;
        --suffix_prompt) suffix_prompt="$2"; shift 2 ;;
        --timeout) timeout="$2"; shift 2 ;;
        --seed) seed="$2"; shift 2 ;;
        --mem_fraction_static) mem_fraction_static="$2"; shift 2 ;;
        --port) port="$2"; shift 2 ;;
        --eval_data_path) eval_data_path="$2"; shift 2 ;;
        --config_path) config_path="$2"; shift 2 ;;
    esac
done

if [ -z "$exp_name" ]; then
    exp_name="$(basename "$(dirname "$model_path")")_$(basename "$model_path")"
fi

log_dir="logs"
mkdir -p $log_dir

uid="$(date +%Y%m%d_%H%M%S)"

python src/eval/inference.py -u "model_path=${model_path},tp=${tp},dp=${dp},mem_fraction_static=${mem_fraction_static},port=${port}" --only_start_server 

python src/eval/inference.py -u \
"\
suffix_prompt=${suffix_prompt},\
prefix_prompt=${prefix_prompt},\
eval_data_path=${eval_data_path},\
output_dir=${output_dir},\
model_path=${model_path},\
exp_name=${exp_name},\
max_new_tokens=${max_new_tokens},\
batch_size=${batch_size},\
temperature=${temperature},\
limit=${limit},\
overwrite=${overwrite},\
tokenizer_path=${tokenizer_path},\
timeout=${timeout},\
seed=${seed},\
mem_fraction_static=${mem_fraction_static},\
port=${port},\
dp=${dp},\
tp=${tp}\
" \
-c ${config_path} \
--only_inference > ${log_dir}/inference_${uid}.log 2>&1

pkill -f sglang || true
sleep 5

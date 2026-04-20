
# MedCoach

MedCoach is a concise hierarchical distillation framework designed to enhance the reasoning capabilities of lightweight language models on complex medical tasks. It leverages knowledge graph augmentation and a phased chain-of-thought distillation process to improve factual reliability and multi-step reasoning.


## Setup & Data Preparation

### 1. Prepare Knowledge Graph Embeddings

Download the PrimeKG knowledge graph file (`kg.csv`) and compute medical embeddings for each entry, then store them in the vector database.

```bash
make -f exps/pipeline.makefile kg_prepare
```

### 2. Construct Distillation Data

Build and format the datasets required for the three-stage distillation process.

```bash
make -f exps/pipeline.makefile data \
    GPU_COUNT="x" \
    MODE=online \
    DATASETS="m1kself" \
    ONLINE_TEACHER_MODEL_NAME="deepseek-reasoner" \
    ONLINE_MODEL_NAME="deepseek-chat" \
    GENERATION_PARAMS='{"max_tokens": 4000}' \
    BACKEND_PARAMS='{"base_url":"https://api.deepseek.com/v1","require_all_responses":false,"}' \
    TIMESTAMP=20250000-0-0 
```

## Training Stages

### Stage 1: Sub-Question-Level Fine-Tuning
Train the student model to solve fine-grained sub-questions with integrated knowledge support.

```bash
bash exps/sft_deepspeed.sh \
    --model_name "xxxx" \
    --train_dataset_name "xxxx" \
    --epochs 1 \
    --lr 1e-6 \
    --global_batch_size 128
```

### Stage 2: Knowledge-Aware Preference Optimization

Improve knowledge discrimination by contrasting valid response with adversarially perturbed negative samples.

```bash
bash exps/dpo_deepspeed.sh \
    --model_name "xxxx" \
    --train_file_path "xxxx" \
    --epochs 1 \
    --lr 5e-7 \
    --beta 0.1 \
    --global_batch_size 16 
```

### Stage 3: Chain-Level Fine-Tuning 

Enable coherent global reasoning by fine-tuning on full knowledge-enhanced reasoning chains.

```bash
 bash exps/sft_deepspeed.sh \
    --model_name "xxxx" \
    --train_dataset_name "xxxx" \
    --epochs 5 \
    --lr 1e-5 \
    --weight_decay 1e-4 \
    --global_batch_size 16 
```


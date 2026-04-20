
MODE ?= online
ONLINE_MODEL_NAME ?= deepseek-reasoner
ONLINE_TEACHER_MODEL_NAME ?= ${ONLINE_MODEL_NAME}
GENERATION_PARAMS ?= ''
BACKEND_PARAMS   ?= {"max_requests_per_minute":100,"max_tokens_per_minute":100000}

DATA_GEN_MODEL   ?= Qwen/Qwen3-8B
PREF_GEN_MODEL   ?= $(DATA_GEN_MODEL)

DATASETS         ?= m1kself
NUM_SAMPLES      ?= 

GPU_COUNT		 ?= 4 

SEED             ?= 42 

OUTPUT_BASE      ?= outputs
OUTPUT_ROOT      ?= outputs/grounding
TIMESTAMP 		 ?= $(shell date +%Y%m%d-%H%M%S-%N) # Need be provided by user
LOG_DIR		 	 ?= logs/$(TIMESTAMP)

$(shell mkdir -p $(LOG_DIR))

ifeq ($(NUM_SAMPLES),)
	COMBINED_NAME = $(DATASETS)
else
	COMBINED_NAME = $(DATASETS)_n$(NUM_SAMPLES)
endif

ifeq ($(MODE),online)
	MODEL_FOLDER := $(ONLINE_TEACHER_MODEL_NAME)
	ifneq ($(ONLINE_MODEL_NAME),$(ONLINE_TEACHER_MODEL_NAME))
		DATA_DIR := $(OUTPUT_ROOT)/$(MODEL_FOLDER)/$(COMBINED_NAME)/$(ONLINE_MODEL_NAME)
	else
		DATA_DIR := $(OUTPUT_ROOT)/$(MODEL_FOLDER)/$(COMBINED_NAME)
	endif
	else
	MODEL_FOLDER := $(shell basename $(DATA_GEN_MODEL))
	DATA_DIR := $(OUTPUT_ROOT)/$(MODEL_FOLDER)/$(COMBINED_NAME)
endif
DATA_ROOT_FOR_ORIG := $(OUTPUT_ROOT)/$(MODEL_FOLDER)/$(COMBINED_NAME)

FINAL_JSONL      := $(DATA_DIR)/final.jsonl
ORIG_JSONL       := $(DATA_ROOT_FOR_ORIG)/orig.jsonl

SUBQ_REWRITE_JSONL := $(DATA_DIR)/subq_rewrite.jsonl
CHAIN_REWRITE_JSONL  := $(DATA_DIR)/chain_rewrite.jsonl

PREF_DATA_JSONL  := $(DATA_DIR)/perturb_pref.jsonl

PYTHON := python

SHELL := /usr/bin/bash
.ONESHELL:
.SHELLFLAGS := -eu -o pipefail -c


.PHONY: all data kg_prepare 

all: kg_prepare data

kg_prepare: 
	python src/data_curation/embed_kg.py \
	2>&1 | tee $(LOG_DIR)/embed_kg.log

data: $(PREF_DATA_JSONL)


# Step 1: Generate grounded subquestions (final.jsonl) and mcq responses (orig.jsonl)
$(FINAL_JSONL) $(ORIG_JSONL):
	@echo ">>> [Data Generation] Step 1: Running run_pipeline.py to generate subquestions and original responses..."
	mkdir -p $(DATA_DIR)
	$(PYTHON) src/data_curation/run_pipeline.py \
		--model_path $(DATA_GEN_MODEL) \
		--dataset_names "$(DATASETS)" \
		--output_dir $(OUTPUT_ROOT) \
		--dp_size ${GPU_COUNT} \
		--seed $(SEED) \
		--mode $(MODE) \
		--online_model_name $(ONLINE_MODEL_NAME) \
		--online_teacher_model_name $(ONLINE_TEACHER_MODEL_NAME) \
		--generation_params '$(GENERATION_PARAMS)' \
		--backend_params '$(BACKEND_PARAMS)' \
		$(if $(NUM_SAMPLES),--num_samples $(NUM_SAMPLES)) 2>&1 | tee $(LOG_DIR)/run_pipeline.log

# Step 2: KG rewriting (subq_rewrite.jsonl, chain_rewrite.jsonl)
$(SUBQ_REWRITE_JSONL) $(CHAIN_REWRITE_JSONL): $(FINAL_JSONL)
	@echo ">>> [Data Generation] Step 2: Running run_embed_subq_kg.py for knowledge rewriting..."
	$(PYTHON) src/data_curation/run_embed_subq_kg.py \
		--input $(FINAL_JSONL) \
		--mode $(MODE) \
		--dp_size ${GPU_COUNT} \
		--online_model_name $(ONLINE_MODEL_NAME) \
		--generation_params '$(GENERATION_PARAMS)' \
		--backend_params '$(BACKEND_PARAMS)' \
		--model_path $(DATA_GEN_MODEL) 2>&1 | tee $(LOG_DIR)/run_embed_subq_kg.log

# Step 3: Generate preference data (perturb_pref.jsonl)
$(PREF_DATA_JSONL): $(SUBQ_REWRITE_JSONL)
	mkdir -p $(dir $@)
	@echo ">>> [Data Generation] Step 3: Running run_perturb_pref.py to generate preference data..."
	$(PYTHON) src/data_curation/run_perturb_pref.py \
		--input $(SUBQ_REWRITE_JSONL) \
		--output $(PREF_DATA_JSONL) \
		--mode $(MODE) \
		--dp_size ${GPU_COUNT} \
		--online_model_name $(ONLINE_MODEL_NAME) \
		--generation_params '$(GENERATION_PARAMS)' \
		--backend_params '$(BACKEND_PARAMS)' \
		--model_path $(PREF_GEN_MODEL) 2>&1 | tee $(LOG_DIR)/run_perturb_pref.log

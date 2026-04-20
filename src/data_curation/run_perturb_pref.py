from collections import defaultdict
import dotenv
dotenv.load_dotenv()
import logging
import re
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    stream=sys.stdout
)

logger = logging.getLogger(__name__)

from tqdm import tqdm
import os
import pickle
import argparse, json, gc
import pandas as pd
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from datasets import Dataset

import openai
from transformers import AutoTokenizer
from core.sglang_server import SGLangServer
from core.batch_processor import BatchProcessor
import importlib
import torch
import random

from utils.kg_utils import retrieve_topk_kg
from triplet2text_dict import relation_templates, negative_relation_templates

PERTURB_LIMIT_ENABLED = True
PERTURB_SAMPLING_ENABLED = True
MAX_PERTURB_SAMPLES   = 1000
MAX_SAMPLES_PER_PROMPT = 2
SEED                  = 42

def main():
    parser = argparse.ArgumentParser(description="Generate preference data: prompt/chosen/rejected")
    parser.add_argument("--input",   default="outputs/grounding/Qwen3-8B/pubmedqa/subq_rewrite.jsonl")
    parser.add_argument("--output", default=None, help="Output preference data JSONL")
    parser.add_argument("--model_path", default="Qwen/Qwen3-8B")
    parser.add_argument("--port",    type=int, default=29990)
    parser.add_argument("--tp_size", type=int, default=1)
    parser.add_argument("--dp_size", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument(
        "--kg_meta_map_path",
        default="data/kg_meta_map.pkl",
        help="Pre-built and cached kg_meta_map pickle path"
    )
    parser.add_argument(
        "--entity_pool_path",
        default="data/entity_pool.pkl",
        help="Pre-built and cached entity_pool pickle path"
    )
    parser.add_argument(
    "--max_irrelevant", type=int, default=10,
    help="Maximum number of irrelevant_kg to use for perturbation per sample"
    )
    parser.add_argument("--mode", default="local", choices=["local","online"])
    parser.add_argument("--online_model_name", default="deepseek-r1", help="Online model name")
    parser.add_argument(
        "--generation_params", default="", 
        help="generation_params in online mode, JSON string, can be left blank to use TaskLLM default"
    )
    parser.add_argument(
        "--backend_params", 
        default='{"max_requests_per_minute":100,"max_tokens_per_minute":100000}',
        help="backend_params in online mode, JSON string, api_key is always read from environment variables"
    )
    
    args = parser.parse_args()

    mode = args.mode

    assert os.path.exists(args.input), f"Input file does not exist: {args.input}"

    if args.output is None:
        args.output = args.input.replace("subq_rewrite.jsonl", "perturb_pref.jsonl")


    logger.info(f"[KG] Retrieval results already exist, skipping retrieval and loading directly: {args.input}")
    records = []
    with open(args.input, 'r', encoding='utf-8') as _f:
        for _line in _f:
            _line = _line.strip()
            if not _line:
                continue
            records.append(json.loads(_line))
    ds_retr = Dataset.from_list(records)
    if args.num_samples:
        logger.info(f"Taking first {args.num_samples} samples")
        ds_retr = ds_retr.select(range(args.num_samples))

    logger.info(f"Number of samples to perturb: {len(ds_retr)}")

    with open(args.kg_meta_map_path, "rb") as f:
        kg_meta_map = pickle.load(f)
    logger.info(f"Loaded kg_meta_map from {args.kg_meta_map_path}, entries={len(kg_meta_map)}")

    with open(args.entity_pool_path, "rb") as f:
        entity_pool = pickle.load(f)
    logger.info(f"Loaded entity_pool from {args.entity_pool_path}, types={list(entity_pool.keys())}")

    ds_retr = ds_retr.filter(
        lambda x: x.get("relevant_kg") and any(x.get("relevant_flags", [])),
        load_from_cache_file=False
    )
    logger.info(
        f"Remaining samples after filtering: {len(ds_retr)} "
    )

    if PERTURB_LIMIT_ENABLED and len(ds_retr) > MAX_PERTURB_SAMPLES:
        ds_retr = ds_retr.shuffle(seed=SEED).select(range(MAX_PERTURB_SAMPLES))
        logger.info(f"Sampling before perturbation: {len(records)} → {len(ds_retr)} records after sampling")

    # Add pre-sampling by original question
    if PERTURB_SAMPLING_ENABLED:
        # Group by original question and sample
        records_list = list(ds_retr)
        grouped = defaultdict(list)
        for rec in records_list:
            prompt_key = rec.get("prompt", "")
            grouped[prompt_key].append(rec)
        
        # Sample from each group
        sampled_records = []
        for prompt_group in grouped.values():
            if len(prompt_group) > MAX_SAMPLES_PER_PROMPT:
                sampled_records.extend(random.sample(prompt_group, MAX_SAMPLES_PER_PROMPT))
            else:
                sampled_records.extend(prompt_group)
        
        ds_retr = Dataset.from_list(sampled_records)
        logger.info(f"Remaining samples after sampling by original question: {len(ds_retr)}")

    def augment_neg(item: dict) -> dict:
        orig   = item["top_kg"]
        flags  = item["relevant_flags"]
        rel_kg_set = {f"<{kg['x_name']}, {kg['display_relation']}, {kg['y_name']}>" 
            for kg, is_rel in zip(orig, flags) if is_rel}
        #irr_all = [kg["kg_sentence"] for kg,f in zip(orig,flags) if not f]
        irr_all = [f"<{kg['x_name']}, {kg['display_relation']}, {kg['y_name']}>" for kg,f in zip(orig,flags) if not f]
        # Limit quantity
        neg_irrelevant = random.sample(
            irr_all, min(len(irr_all), args.max_irrelevant)
        )
        neg_swapped    = []
        neg_negation   = []

        for kg, is_rel in zip(orig, flags):
            if not is_rel:
                continue
            #sent = kg["kg_sentence"]
            #sent = f"({kg['x_name']}, {kg['display_relation']}, {kg['y_name']})"
            #row = kg_meta_map[sent]
            row = kg_meta_map[kg["kg_sentence"]]
            rel          = row["relation"]
            disp         = row["display_relation"]
            xt           = row["x_type"]
            yt           = row["y_type"]
            x_name       = row["x_name"]
            y_name       = row["y_name"]

            type1, type2 = sorted([xt, yt])
            tpl_key = (rel, disp, type1, type2)
            tpl_key_rev = (rel, disp, yt, xt)  
            
            if tpl_key in relation_templates:
                tpl_pos = relation_templates[tpl_key]
                subject_name, object_name = x_name, y_name
            elif tpl_key_rev in relation_templates:
                tpl_pos = relation_templates[tpl_key_rev]
                subject_name, object_name = y_name, x_name  
            else:
                logger.warning(f"No positive template for key {tpl_key} or {tpl_key_rev}")
                continue

            actual_subject_type = xt if subject_name == x_name else yt
            actual_object_type = yt if object_name == y_name else xt

            cands_object = entity_pool[actual_object_type]
            if cands_object:
                valid_candidates = [obj for obj in cands_object 
                                if f"<{subject_name}, {disp}, {obj}>" not in rel_kg_set]
                if valid_candidates:
                    n_object_name = random.choice(valid_candidates)
                    swapped_triple = tpl_pos.format(subject=subject_name, object=n_object_name)
                    neg_swapped.append(swapped_triple)

            cands_subject = entity_pool[actual_subject_type]
            if cands_subject:
                valid_candidates = [subj for subj in cands_subject 
                                if f"<{subj}, {disp}, {object_name}>" not in rel_kg_set]
                if valid_candidates:
                    n_subject_name = random.choice(valid_candidates)
                    swapped_triple = tpl_pos.format(subject=n_subject_name, object=object_name)
                    neg_swapped.append(swapped_triple)

            if tpl_key in negative_relation_templates:
                tpl_neg = negative_relation_templates[tpl_key]
                neg_negation.append(tpl_neg.format(subject=x_name, object=y_name))
            elif tpl_key_rev in negative_relation_templates:
                tpl_neg = negative_relation_templates[tpl_key_rev]
                neg_negation.append(tpl_neg.format(subject=y_name, object=x_name))
            else:
                logger.warning(f"No negative template for key {tpl_key} or {tpl_key_rev}")

        item["irrelevant_kg"] = [{"kg_sentence": s} for s in neg_irrelevant]
        item["swapped_kg"]    = [{"kg_sentence": s} for s in neg_swapped]
        item["negated_kg"]    = [{"kg_sentence": s} for s in neg_negation]
        return item

    ds_retr = ds_retr.map(
        augment_neg,
        batched=False,
        num_proc=min(4, os.cpu_count()),
    )

    server = None
    if mode == 'local':
        server = SGLangServer(model_path=args.model_path,
                            port=args.port, tp_size=args.tp_size,
                            dp_size=args.dp_size,)
        server.start()
        client    = openai.Client(base_url=f"http://127.0.0.1:{args.port}/v1", api_key="EMPTY")
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
        processor = BatchProcessor(client, tokenizer,
                                batch_size=args.batch_size,
                                max_new_tokens=args.max_new_tokens)
    else:
        #os.environ["CURATOR_DISABLE_CACHE"] = "1"
        online_model_name = args.online_model_name
        if args.generation_params.strip():
            gen_params = json.loads(args.generation_params)
        else:
            gen_params = None
        backend_params = json.loads(args.backend_params) 

        if backend_params["base_url"] == "https://dashscope.aliyuncs.com/compatible-mode/v1":
            pass
            #backend_params["api_key"] = os.environ.get("OPENAI_API_KEY")
        elif backend_params["base_url"] == "https://api.deepseek.com/v1":
            os.environ["OPENAI_API_KEY"] = os.environ["DEEPSEEK_API_KEY"]
            #backend_params["api_key"] = os.environ.get("DEEPSEEK_API_KEY")
        else:
            raise ValueError(f"Unknown base_url: {backend_params['base_url']}, please check backend_params")
        
        # Build kwargs for TaskLLM init
        llm_kwargs = {
            "model_name": online_model_name,
            "backend_params": backend_params
        }
        if gen_params is not None:
            llm_kwargs["generation_params"] = gen_params


    try:
        all_recs = []
        for name, mod in [
            ("irrelevant", "tasks.perturb_irrelevant"),
            ("swapped",    "tasks.perturb_swapped"),
            ("negation",   "tasks.perturb_negation"),
        ]:
            logger.info(f"--- Running {name} perturbation ---")
            
            if mode == 'local':
                task = importlib.import_module(mod).Task()
                ds_task  = processor.process_dataset(ds_retr, task)
            else:
                taskLLM = importlib.import_module(mod).TaskLLM(**llm_kwargs)
                ds_task = taskLLM(ds_retr)

            out_fp     = args.output.replace(".jsonl", f"_{name}.jsonl")
            recs, dbg  = [], []

            for rec in ds_task:
                dbg.append(rec)   # full record

            with open(out_fp, "w", encoding="utf-8") as fo:
                valid_count = 0
                for rec in dbg:
                    context = rec.get("context", "").strip()
                    user_prompt = rec["prompt"].strip()
                    if context:
                        final_prompt = f"Context:\n{context}\n\nQuestion:\n{user_prompt}"
                    else:
                        final_prompt = user_prompt

                    rejected_text = rec["rejected"].strip()
                    chosen_text = rec["chosen"].strip()
                    if not rejected_text or rejected_text == chosen_text:
                        continue
                    
                    rd = {
                        "prompt": final_prompt,
                        "chosen": chosen_text,
                        "rejected": rejected_text
                    }
                    fo.write(json.dumps(rd, ensure_ascii=False) + "\n")
                    recs.append(rd)
                    valid_count += 1
            all_recs.extend(recs)

            logger.info(f"{name} valid samples: {valid_count}, total samples: {len(dbg)}")


        uni_fp = args.output
        with open(uni_fp, "w", encoding="utf-8") as fu:
            for rd in all_recs:
                fu.write(json.dumps(rd, ensure_ascii=False) + "\n")
        logger.info(f"Unified perturbation file written → {uni_fp}")
    finally:
        if mode == 'local':
            server.terminate()
            torch.cuda.empty_cache()
            gc.collect()

if __name__ == "__main__":
    main()

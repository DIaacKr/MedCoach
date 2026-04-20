import dotenv
dotenv.load_dotenv()
import logging
import os
import re
import sys
logging.basicConfig(
    level=logging.INFO, #
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    stream=sys.stdout
)

logger = logging.getLogger(__name__)

from datasets import Dataset
from collections import defaultdict
import argparse, json
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from datasets import load_dataset, concatenate_datasets
from tqdm import tqdm

import sys, torch, gc, importlib
import openai
from core.sglang_server import SGLangServer
from core.batch_processor import BatchProcessor
from transformers import AutoTokenizer

from utils.kg_utils import retrieve_topk_kg

def main():
    parser = argparse.ArgumentParser(
        description="Embedding the sub-problem grounded_text of the run_pipeline output and retrieving topK in the KG index."
    )
    parser.add_argument(
        "--input",
        default="outputs/grounding/Qwen3-8B/pubmedqa/final.jsonl",
    )
    parser.add_argument(
        "--kg_index",
        default="data/kg_index.faiss",
    )
    parser.add_argument(
        "--kg_meta",
        default="data/kg_metadata.parquet",
    )
    parser.add_argument(
        "--embed_model",
        default="abhinand/MedEmbed-large-v0.1",
    )
    parser.add_argument(
        "--top_k", type=int, default=100, #
    )
    parser.add_argument(
        "--num_samples", type=int, default=None,
    )
    parser.add_argument(
        "--output",
        default=None,
    )
    parser.add_argument("--model_path", default="/home/lc/models/RAW/7B-level/general/Qwen3-8B")
    parser.add_argument("--port",        type=int, default=29990)
    parser.add_argument("--tp_size",     type=int, default=1)
    parser.add_argument("--dp_size",     type=int, default=1)
    parser.add_argument("--batch_size",  type=int, default=256)
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--mode", default="local", choices=["local","online"])
    parser.add_argument("--online_model_name", default="deepseek-r1")
    parser.add_argument(
        "--generation_params", default="", 
    )
    parser.add_argument(
        "--backend_params", 
        default='{"max_requests_per_minute":100,"max_tokens_per_minute":100000}',
    )

    args = parser.parse_args()

    mode = args.mode

    records = []
    with open(args.input, 'r', encoding='utf-8') as _f:
        for _line in _f:
            _line = _line.strip()
            if not _line:
                continue
            records.append(json.loads(_line))
    ds = Dataset.from_list(records)
    logger.info(f"Loaded sub-question dataset {args.input}, size = {len(ds)}")

    if args.output is None:
        args.output = os.path.join(os.path.dirname(args.input), "subq_topkg.jsonl")

    ds_filtered = ds.filter(
        lambda ex: ex.get("subquestion","").strip() != "" 
                   and ex.get("grounded_text","").strip() != "" 
                   and ex.get("exact_match", False),
        batched=False
    )

    logger.info(f"Filtered sub-question dataset ds_filtered size = {len(ds_filtered)}")
    
    if args.num_samples and args.num_samples>0:
        ds_filtered = ds_filtered.select(range(min(args.num_samples, len(ds_filtered))))
        logger.info(f"Only process first {args.num_samples} samples, ds_filtered size = {len(ds_filtered)}")

    if os.path.exists(args.output):
        logger.info(f"[KG] Retrieval results already exist, skip retrieval and load directly: {args.output}")
        # Manually read output JSONL to avoid type inconsistency caused by datasets.load_dataset chunk inference
        records = []
        with open(args.output, "r", encoding="utf-8") as _f:
            for _line in _f:
                _line = _line.strip()
                if not _line:
                    continue
                records.append(json.loads(_line))
        ds_retr = Dataset.from_list(records)
    else:
        ds_retr = retrieve_topk_kg(ds_filtered, args.output,args.kg_index, args.kg_meta,args.embed_model, args.top_k)
        ds_retr = ds_retr.filter(
            lambda ex: len(ex.get("top_kg", [])) > 0,
            batched=False
        )

    logger.info(f"Retrieved top-k KG, ds_retr size = {len(ds_retr)}")

    server = None
    if mode == 'local':
        # 1. Start SGLang server
        server = SGLangServer(
            model_path=args.model_path,
            port=args.port,
            tp_size=args.tp_size,
            dp_size=args.dp_size,
            seed=42,
            mem_fraction_static=0.85
        )
        server.start()
        client    = openai.Client(base_url=f"http://127.0.0.1:{args.port}/v1", api_key="EMPTY")
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
        processor = BatchProcessor(client, tokenizer, args.batch_size, args.max_new_tokens)

        relevance_task = importlib.import_module("tasks.judge_kg_relevance").Task()        
        rewrite_task = importlib.import_module("tasks.rewrite_with_kg").Task()
    else:
        os.environ["CURATOR_DISABLE_CACHE"] = "1"
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

        judgeLLM = importlib.import_module("tasks.judge_kg_relevance").TaskLLM(**llm_kwargs)
        subqrewriteLLM = importlib.import_module("tasks.rewrite_with_kg").TaskLLM(**llm_kwargs)

    try:

        if mode == 'local':
            ds_rel_all = processor.process_dataset(ds_retr, relevance_task)
        else:
            ds_rel_all = judgeLLM(ds_retr)

        def mark_flags(item: dict) -> dict:
            top_kg = item["top_kg"]
            rel    = item["relevant_kg"]

            rel_sents = {r for r in rel}
            flags = [f"<{kg['x_name']}, {kg['display_relation']}, {kg['y_name']}>" in rel_sents for kg in top_kg]

            item["relevant_flags"] = flags
            return item

        ds_rel_all = ds_rel_all.map(mark_flags, batched=False)
        
        ds_rel_valid = ds_rel_all.filter(lambda ex: any(ex.get("relevant_flags", [])), batched=False)
        logger.info(f"Have relevant KG = {len(ds_rel_valid)}")

        logger.info(f"But all go into rewrite")

        if mode == 'local':
            ds_rewrite = processor.process_dataset(ds_rel_all, rewrite_task)
        else:
            ds_rewrite = subqrewriteLLM(ds_rel_all)

        flags   = ds_rewrite["modified"]
        total   = len(flags)
        changed = sum(flags)
        logger.info(f"Total entries: {total}, rewrite effective: {changed} ({changed/total*100:.2f}%)")

        # Save sub-question rewrite results
        subq_rewrite_out = args.output.replace("subq_topkg.jsonl", "subq_rewrite.jsonl")
        ds_rewrite.to_json(subq_rewrite_out, orient="records", lines=True, force_ascii=False)
        logger.info(f"Sub-question rewrite results JSONL written to: {subq_rewrite_out}")

        # Filter out entries without modification
        ds_rewrite_filtered = ds_rewrite.filter(
            lambda ex: ex.get("modified", False),
            batched=False
        )
        logger.info(f"Rewrite completed, ds_rewrite_filtered size = {len(ds_rewrite_filtered)}")

        logger.info("But still keep unrevised records")

        records = list(ds_rewrite)
        logger.info(f"Records before grouping = {len(records)}")
        groups = defaultdict(list)
        for rec in records:
            groups[rec["prompt"]].append(rec)
        logger.info(f"After grouping by prompt, total {len(groups)} groups")

        outputs = []
        for prompt, items in groups.items():
            orig_mr = items[0]["mcq_model_response"]
            m = re.search(r'<think>[\s\S]*?</think>([\s\S]*)', orig_mr, re.IGNORECASE)
            partial_orig = m.group(1).strip() if m else orig_mr

            if partial_orig == "":
                logger.warning(f"Original reply partial_response is empty, skip: {prompt}")
                raise ValueError(f"error")

            original = items[0]["think_content"]
            segs = []
            for it in items:
                if it["modified"]:
                    subq = it["subquestion"].strip()
                    gt = it["grounded_text"]
                    rt = it["rewritten_text"]

                    rt = rt.strip() #
                    if not gt:
                        continue
                    idx = original.find(gt)
                    if idx < 0:
                        logger.warning(f"grounded_text not found in original reply, skip: {gt}")
                        continue
                    segs.append((idx, idx + len(gt), rt))

            new_tc = original
            if segs:
                segs.sort(key=lambda x: x[0])
                merged = []
                cur_s, cur_e, cur_ts = segs[0][0], segs[0][1], [segs[0][2]]
                for s, e, t in segs[1:]:
                    if s <= cur_e:
                        cur_e = max(cur_e, e)
                        cur_ts.append(t)
                    else:
                        merged.append((cur_s, cur_e, "\n".join(cur_ts)))
                        cur_s, cur_e, cur_ts = s, e, [t]
                merged.append((cur_s, cur_e, "\n".join(cur_ts)))

                parts, last = [], 0
                for s, e, txt in merged:
                    parts.append(original[last:s])
                    parts.append(txt)
                    last = e
                parts.append(original[last:])
                new_tc = "".join(parts)

            outputs.append({
                "prompt": prompt,
                "think_content_rewritten": new_tc,
                "partial_response_before": partial_orig,
            })
        logger.info(f"think_content replacement completed, outputs size = {len(outputs)}")

        ds_rest = Dataset.from_list(outputs)
        logger.info(f"Construct final output ds_rest size = {len(ds_rest)}")

        final_rewrite_out = args.output.replace("subq_topkg.jsonl", "chain_rewrite.jsonl")
        ds_rest.to_json(final_rewrite_out,
                        orient="records", lines=True, force_ascii=False)
        logger.info(f"Final MCQ rewrite results JSONL written to: {final_rewrite_out}")

    finally:
        if mode == 'local':
            server.terminate()
            torch.cuda.empty_cache()
            gc.collect()

if __name__ == "__main__":
    main()

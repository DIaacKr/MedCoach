import dotenv
dotenv.load_dotenv()
from collections import defaultdict
import json
import os
import click, sys, logging


logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)],
                    force=True)

logger = logging.getLogger(__name__)


import torch, gc, importlib
from pathlib import Path
from transformers import AutoTokenizer
import openai
from datasets import Dataset, concatenate_datasets

from core.sglang_server import SGLangServer
from core.batch_processor import BatchProcessor
from utils.dataset_utils import dataset_loaders


@click.command()
@click.option("--model_path", required=True, help="Model path")
@click.option("--dataset_names", default="medqa", help="Underscore-separated dataset names")
@click.option("--output_dir", default="./outputs/grounding", help="Output root directory")
@click.option("--port", default=29990, type=int, help="SGLang port")
@click.option("--tp_size", default=1, type=int, help="Tensor parallel size")
@click.option("--dp_size", default=1, type=int, help="Data parallel size")
@click.option("--batch_size", default=256, type=int, help="Batch size")
@click.option("--max_new_tokens", default=4096, type=int, help="Maximum generation length")
@click.option("--num_samples", default=None, type=int, help="Number of test samples")
@click.option("--seed", default=42, type=int, help="Random seed")
@click.option("--mem_fraction_static", default=0.85, type=float, help="Memory fraction")
@click.option("--mode", default="local", type=click.Choice(["local", "online"]))
@click.option("--online_model_name", default="deepseek-r1", help="Online model name")
@click.option("--online_teacher_model_name", default="", help="Online teacher model name, if empty use online_model_name")
@click.option(
    "--generation_params",
    default='',
)
@click.option(
    "--backend_params",
    default='{"max_requests_per_minute": 100, "max_tokens_per_minute": 100000}',
)
def main(**cfg):
    output_root = Path(cfg["output_dir"])
    output_root.mkdir(parents=True, exist_ok=True)
    mode = cfg["mode"]
    teacher_model_name = cfg["online_teacher_model_name"] or cfg["online_model_name"]
    online_model_name = cfg["online_model_name"]

    names = [n.strip() for n in cfg["dataset_names"].split("_") if n.strip()]
    ds_list = []
    for name in names:
        if name not in dataset_loaders:
            logger.warning(f"Dataset {name} not found, skipping")
            continue
        ds_tmp = dataset_loaders[name](verbose=False)
        if cfg["num_samples"]:
            ds_tmp =  ds_tmp.shuffle(seed=cfg["seed"]).select(range(cfg["num_samples"]))
        ds_list.append(ds_tmp)
    if not ds_list:
        logger.error("No available datasets, exiting")
        sys.exit(1)
    ds = concatenate_datasets(ds_list)
    combined_name = "_".join(names)
    if cfg["num_samples"]:
        combined_name += f"_n{cfg['num_samples']}"
    logger.info(f"Merged dataset {combined_name} size = {ds.num_rows}")

    if mode == "local":
        model_dir = output_root / Path(cfg["model_path"]).name / combined_name
        coach_dir = model_dir #
    else:
        model_dir = output_root / Path(teacher_model_name).name / combined_name #
        if online_model_name != teacher_model_name:
            coach_dir = model_dir / Path(online_model_name).name
        else:
            coach_dir = model_dir
    model_dir.mkdir(parents=True, exist_ok=True)
    coach_dir.mkdir(parents=True, exist_ok=True)


    server = None
    if mode == "local":
        mcq = importlib.import_module("tasks.mcq_evaluation").Task()
        decomp = importlib.import_module("tasks.question_decomposition").Task()
        ground = importlib.import_module("tasks.grounding").Task()
        context_task = importlib.import_module("tasks.context_extraction").Task()

        server = SGLangServer(model_path=cfg["model_path"],
                            port=cfg["port"],
                            tp_size=cfg["tp_size"],
                            dp_size=cfg["dp_size"],
                            seed=cfg["seed"],
                            mem_fraction_static=cfg["mem_fraction_static"])
        server.start()

        client = openai.Client(base_url=f"http://127.0.0.1:{cfg['port']}/v1", api_key="EMPTY")
        tokenizer = AutoTokenizer.from_pretrained(cfg["model_path"])
        processor = BatchProcessor(client, tokenizer, cfg["batch_size"], cfg["max_new_tokens"])
    else:
        os.environ["CURATOR_DISABLE_CACHE"] = "1"
        online_model_name = cfg["online_model_name"]
        if cfg["generation_params"].strip():
            gen_params = json.loads(cfg["generation_params"])
        else:
            gen_params = None
        backend_params = json.loads(cfg["backend_params"]) 
        
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

        teacher_llm_kwargs = {
            "model_name": teacher_model_name,
            "backend_params": backend_params
        }
        if gen_params is not None:
            teacher_llm_kwargs["generation_params"] = gen_params
        
        out1 = model_dir / "orig.jsonl"
        if out1.exists():
            pass
        else:
            mcqLLM = importlib.import_module("tasks.mcq_evaluation").TaskLLM(**teacher_llm_kwargs)
        decompLLM = importlib.import_module("tasks.question_decomposition").TaskLLM(**llm_kwargs)
        groundLLM = importlib.import_module("tasks.grounding").TaskLLM(**llm_kwargs)
        context_taskLLM = importlib.import_module("tasks.context_extraction").TaskLLM(**llm_kwargs)

    try:
        logger.info(f"=== MCQ Evaluation ===")
        
        out1 = model_dir / "orig.jsonl"
        if out1.exists():
            logger.info(f"{out1} already exists, skipping MCQ evaluation and loading directly")
            records = []
            with open(out1, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        records.append(json.loads(line))
            ds1 = Dataset.from_list(records)
        else:
            if mode == "local":
                ds1 = processor.process_dataset(ds, mcq)
            else:
                ds1 = mcqLLM(ds)

            num_before = ds1.num_rows
            ds1 = ds1.filter(lambda ex: ex["think_content"] != "", batched=False)
            num_after = ds1.num_rows
            logger.info(f"Filtered {num_before - num_after} records without think_content, retained {num_after} records")

            # Save ds1 (original questions + MCQ answers) for subsequent SFT
            ds1.to_json(out1, orient="records", lines=True, force_ascii=False)
            logger.info(f"Original questions + MCQ answers saved to {out1}")


        logger.info(f"=== Context Extraction ===")
        if mode == "local":
            ds_ctx = processor.process_dataset(ds1, context_task)
        else:
            ds_ctx = context_taskLLM(ds1)

        logger.info(f"=== Question Decomposition ===")
        if mode == "local":
            ds2 = processor.process_dataset(ds_ctx, decomp)
        else:
            ds2 = decompLLM(ds_ctx)

        # ===== Flatten sub-questions one by one =====
        records = []
        for idx, example in enumerate(ds2):
                # Keep original fields except subquestions
            base = {k: example[k] for k in example if k != "subquestions"}
            # Mark original sample index for subsequent statistics
            base["orig_index"] = idx
            for sq in example.get("subquestions", []):
                rec = base.copy()
                rec["subquestion"] = sq
                records.append(rec)

        ds2_flat = Dataset.from_list(records)

        logger.info(f"=== Sub-question Grounding ===")
        if mode == "local":
            ds3 = processor.process_dataset(ds2_flat, ground)
        else:
            ds3 = groundLLM(ds2_flat)

        # ===== Check if grounded_text can be fully matched in mcq_model_response =====
        def check_exact(example):
            gt = example.get("grounded_text", "")
            tc = example.get("think_content", "")
            exact = bool(gt and gt in tc)
            return {"exact_match": exact}
        ds3 = ds3.map(check_exact, batched=False)

        # Global statistics
        matches = ds3["exact_match"]
        exact_count = sum(matches)
        total = len(matches)
        rate = (exact_count / total * 100) if total > 0 else 0.0
        logger.info(f"Exact match statistics: {exact_count}/{total}, match rate: {rate:.2f}%")

        # Per original sample sub-question match statistics
        stats = defaultdict(lambda: {"match": 0, "total": 0})
        for ex in ds3:
            idx = ex["orig_index"]
            stats[idx]["total"] += 1
            if ex["exact_match"]:
                stats[idx]["match"] += 1
        for idx, s in stats.items():
            sample_rate = (s["match"] / s["total"] * 100) if s["total"] > 0 else 0.0
            logger.info(f"Sample {idx} sub-question match: {s['match']}/{s['total']}, match rate: {sample_rate:.2f}%")
        out3 = coach_dir / "final.jsonl" #
        ds3.to_json(out3, orient="records", lines=True, force_ascii=False)
        logger.info(f"Sub-questions + grounding results saved to {out3}")
    finally:
        if server:
            server.terminate()
            torch.cuda.empty_cache()
            gc.collect()
        logger.info("Full pipeline completed")

if __name__ == "__main__":
    main()

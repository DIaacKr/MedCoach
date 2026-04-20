import dotenv
dotenv.load_dotenv()

import logging
import sys

logging.basicConfig(
    level=logging.INFO, #
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    stream=sys.stdout
)

logger = logging.getLogger(__name__)

import json
import os
import argparse
from pathlib import Path
from datasets import Dataset

from extract_format import extract_answer
from score import huatuo_match_choice, score

def load_eval_data(file_path, limit=0):
    with open(file_path, "r") as f:
        data = json.load(f)
    
    input_data = []
    if isinstance(data, list):
        data = {"normal": data}

    for k, v in data.items():
        if limit > 0 and len(v) > limit:
            v = v[:limit]
        for da in v:
            da["source"] = k
        input_data.extend(v)
    
    return input_data

def format_prompt(item, prefix_prompt=None):
    option_str = "\n".join([f"{op}. {ans}" for op, ans in item["options"].items()])
    prompt = f"{item['question']}\n{option_str}"
    
    if prefix_prompt:
        prompt = f"{prefix_prompt}\n\n{prompt}"
    
    prompt += "Return your final response within \\boxed{}."
    
    return prompt
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_data_path", required=True)
    parser.add_argument("--output_dir", default="./outputs/remote_eval")
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--prefix_prompt", default="")
    parser.add_argument("--backend_params", 
                       default='{"base_url": "https://api.deepseek.com/v1", "max_requests_per_minute": 5000, "max_tokens_per_minute": 1000000}')
    args = parser.parse_args()

    #os.environ["CURATOR_DISABLE_CACHE"] = "1"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_data = load_eval_data(args.eval_data_path, limit=args.limit)
    print(f"Loaded {len(input_data)} samples")

    backend_params = json.loads(args.backend_params)
    
    if "deepseek" in backend_params.get("base_url", ""):
        os.environ["OPENAI_API_KEY"] = os.environ["DEEPSEEK_API_KEY"]
        logger.info("Using DeepSeek API")

    from bespokelabs import curator
    
    class EvalTask(curator.LLM):
        return_completions_object = True
        def prompt(self, input_sample):
            return format_prompt(input_sample, args.prefix_prompt)
        
        def parse(self, input_sample, response):
            try:
                response_text = response["choices"][0]["message"]["content"] or ""
                option_str = "\n".join([f"{op}. {ans}" for op, ans in input_sample["options"].items()])
                
                extracted_answer = extract_answer(response_text) or ""
                
                huatuo_extracted_answer = huatuo_match_choice(
                    response_text, input_sample["options"]
                ) or ""

                return {
                    **input_sample,
                    "response_text": response_text,
                    "extracted_answer": extracted_answer,
                    "huatuo_extracted_answer": huatuo_extracted_answer,
                    "finish_reason": response["choices"][0].get("finish_reason", ""),
                    "num_gen_tokens": response["usage"].get("completion_tokens", 0) if "usage" in response else 0,
                    "option_str": option_str,
                }
            except Exception as e:
                logger.warning(f"Error parsing response: {e}")
                return {
                    **input_sample,
                    "response_text": "",
                    "extracted_answer": "",
                    "huatuo_extracted_answer": "",
                    "finish_reason": "",
                    "num_gen_tokens": 0,
                    "option_str": "",
                }

    task_llm = EvalTask(
        model_name=args.model_name,
        backend_params=backend_params
    )

    dataset = Dataset.from_list(input_data)

    print("Starting inference...")
    results = task_llm(dataset)
    
    result_list = list(results)
    
    output_path = output_dir / f"{Path(args.eval_data_path).stem}_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result_list, f, indent=2, ensure_ascii=False)
    print(f"Results saved to: {output_path}")

    try:
        metrics, mapped_results = score(result_list)
        
        scored_path = output_path.with_suffix(".scored.json")
        mapped_results.to_json(scored_path, indent=2)
        print(f"Scored results saved to: {scored_path}")
        
        metrics_path = output_dir / "metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        print(f"Metrics saved to: {metrics_path}")
        print(f"Metrics: {metrics}")
        
    except Exception as e:
        print(f"error: {e}")

if __name__ == "__main__":
    main()
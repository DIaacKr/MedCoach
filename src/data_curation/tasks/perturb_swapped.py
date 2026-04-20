import re, json, logging
from bespokelabs import curator
from utils.parsing_utils import extract_json_from_response

logger = logging.getLogger(__name__)

class Task:
    def get_user_prompt(self, item: dict) -> str:
        q    = item["subquestion"]
        a    = item["rewritten_text"]
        swap = item["swapped_kg"]
        
        context = item.get("context", "")
        if context:
            context_block = f"Context:\n{context}\n\n"
        else:
            context_block = ""

        block = "\n".join(f"- {d['kg_sentence']}" for d in swap) or "(none)"
        return (
            f"""
Here is an one-shot example:
Question: How does acetylcholine work in the synapse?
Correct answer: It binds nicotinic and muscarinic receptors.
Swapped triple: Acetylcholine transports oxygen in red blood cells.
Wrong answer: Acetylcholine carries oxygen in red blood cells.

Now generate a WRONG answer for the new data using SWAPPED triples:
Context: 
{context_block if context else "(none)"}
Question: 
{q}
Correct answer: 
{a}
Swapped triples:
{block}

Return ONLY a JSON object with exactly one key "wrong_answer", e.g.:
{{"wrong_answer":"..."}}
"""
        )

    def parse_output(self, item: dict) -> dict:
        raw = item.get("raw_model_response", "").strip()
        
        try:
            obj = extract_json_from_response(raw)
            if obj and "wrong_answer" in obj:
                wrong = obj["wrong_answer"].strip()
                return {
                    "prompt": item["subquestion"],
                    "chosen": item["rewritten_text"],
                    "rejected": wrong,
                    "raw_text": raw
                }
        except Exception as e:
            logger.warning(f"parse_negation JSON loads failed: {e}")
        
        return {
            "prompt": item["subquestion"],
            "chosen": item["rewritten_text"],
            "rejected": "",
            "raw_text": raw
        }

class TaskLLM(curator.LLM):
    return_completions_object = True

    def prompt(self, input_sample):
        return Task().get_user_prompt(input_sample)

    def parse(self, input_sample, response):
        message = response["choices"][0]["message"]
        reasoning_content = message.get("reasoning_content", "")
        content = message["content"]
        
        #logger.info(f"Reasoning content: {reasoning_content}")
        #logger.info(f"Model response content: {content}")

        if reasoning_content:
            raw = f"<think>\n{reasoning_content}\n</think>\n\n{content}"
        else:
            raw = content

        mid = {**input_sample, "raw_model_response": raw}
        new_fields = Task().parse_output(mid)

        result = {**input_sample, **new_fields}
        result.pop("raw_model_response", None)
        return result

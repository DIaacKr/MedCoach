import json
import re
import logging
from bespokelabs import curator

logger = logging.getLogger(__name__)

class Task:

    def get_user_prompt(self, item: dict) -> str:
        think = item.get("think_content", "").strip() or "none"
        prompt = item.get("prompt", "").strip()
        return f"""You are an expert in medical question decomposition.  
Using the reasoning process below, break the original complex question into a sequence of self-contained sub-questions.  
- Each sub-question should correspond to a distinct step or critical content in the reasoning process.  
- Pay special attention to any points of uncertainty or where deeper medical knowledge was invoked.

Reasoning process:
{think}

Original question:
{prompt}

Return ONLY a JSON array of strings, e.g.:
["First self-contained sub-question?", "Second self-contained sub-question?", ...]
"""
    def parse_output(self, item: dict) -> dict:
        response = item.get("raw_model_response", "").strip()
        subqs = []

        parse_text = response
        think_end = re.search(r'</think>', response, re.IGNORECASE)
        if think_end:
            parse_text = response[think_end.end():].strip()

        json_match = re.search(r'(\[[\s\S]*?\])', parse_text)
        if json_match:
            try:
                parsed = json.loads(json_match.group(1))
                if isinstance(parsed, list) and all(isinstance(x, str) for x in parsed):
                    subqs = [s.strip() for s in parsed if s.strip()]
            except json.JSONDecodeError:
                pass

        if not subqs:
            try:
                parsed = json.loads(parse_text)
                if isinstance(parsed, list) and all(isinstance(x, str) for x in parsed):
                    subqs = [s.strip() for s in parsed if s.strip()]
                else:
                    raise ValueError
            except Exception:
                lines = [ln.strip() for ln in parse_text.splitlines() if ln.strip()]
                start = 0
                for idx, ln in enumerate(lines):
                    if re.match(r'^\d+[\.\)]\s+', ln):
                        start = idx
                        break
                for ln in lines[start:]:
                    m = re.match(r'^\d+[\.\)]\s*(.*)', ln)
                    if m:
                        subqs.append(m.group(1).strip())
                    else:
                        break
                
                if not subqs:
                    logger.warning("No valid sub-questions found in the response.")
                    subqs = []  

        return {
            "subquestions": subqs
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

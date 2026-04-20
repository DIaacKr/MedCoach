import json, logging, re
from bespokelabs import curator



class Task:

    def get_user_prompt(self, item: dict) -> str:
        think = item.get("think_content").strip()
        subq = item.get("subquestion").strip()

        final_prompt = f"""
You are an information-retrieval specialist. Given the full reasoning process below and a specific sub-question, locate the single complete sentence or paragraph from the reasoning that answer or are related to the sub-question.

- Preserve the text exactly as it appears: do NOT paraphrase, shorten, or modify punctuation or capitalization.  
- Return ONLY a JSON object with exactly two fields:  
  1. "subquestion": the original sub-question string  
  2. "grounded_text": the exact sentence or paragraph from the reasoning

Reasoning Process:
{think}

Sub-question:
{subq}

Example output:
{{"subquestion": "{subq}", "grounded_text": "<exact sentence or paragraph>"}}
"""
        return final_prompt

    def parse_output(self, item: dict) -> dict:
        raw = item.get("raw_model_response", "").strip()

        think_end = re.search(r'</think>', raw, re.IGNORECASE)
        content = raw[think_end.end():].strip() if think_end else raw

        m = re.search(r'(\{[\s\S]*\})\s*$', content)
        json_str = m.group(1) if m else ""
        obj = {}
        if json_str:
            try:
                obj = json.loads(json_str)
            except json.JSONDecodeError as e:
                logging.warning(f"grounding JSON error: {e}, original JSON: {json_str}")

        if not obj.get("grounded_text"):
            m2 = re.search(r'"grounded_text"\s*:\s*"([^"]*)"', content)
            if m2:
                obj["grounded_text"] = m2.group(1)

        grounded_text = obj.get("grounded_text", "").strip()
        if not grounded_text:
            logging.warning("error")

        return {
            "grounded_text": grounded_text
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

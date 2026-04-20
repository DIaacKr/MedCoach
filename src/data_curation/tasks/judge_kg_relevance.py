import json
import logging
import re
from bespokelabs import curator

logger = logging.getLogger(__name__)

class Task:
    """Judge relevance of each triple in top_kg, directly output list of relevant triples."""

    def get_user_prompt(self, item: dict) -> str:
        subq    = item["subquestion"]
        gt      = item["grounded_text"]
        kg_list = item["top_kg"]
        if kg_list:
            #kg_block = "\n".join(f"- {kg['kg_sentence']}" for kg in kg_list)
            kg_block = "\n".join(f"- <{kg['x_name']}, {kg['display_relation']}, {kg['y_name']}>" for kg in kg_list)
        else:
            kg_block = "(no triples)"
        return (
            "You are a knowledge relevance classifier. "
            "Given a question, its corresponding answer text, and a list of knowledge triples, "
            "select only those triples that are relevant for rewriting the text. "
            f"Question:\n{subq}\n\n"
            f"Corresponding text:\n{gt}\n\n"
            "Knowledge triples (one per line):\n"
            f"{kg_block}\n\n"
            "Return ONLY a JSON object with a single field:\n"
            "{\"relevant_kg\": [<the relevant triples exactly as in the input>]}\n"
            "If none are relevant, return {\"relevant_kg\": []}. "
            "Do not output anything else."
        )

    def parse_output(self, item: dict) -> dict:
        raw = item.get("raw_model_response", "")
        # Clean up possible <think> tags or thinking content from model
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.IGNORECASE|re.DOTALL)
        
        # Clean up possible Markdown code block markers
        raw = re.sub(r'^\s*```(?:json)?\s*', '', raw, flags=re.MULTILINE)
        raw = re.sub(r'\s*```\s*$', '', raw, flags=re.MULTILINE)
        raw = raw.strip()
        
        # Try multiple ways to extract JSON
        json_str = None
        
        # Method 1: Find standard JSON object
        m = re.search(r'(\{[\s\S]*"relevant_kg"[\s\S]*\})', raw)
        if m:
            json_str = m.group(1)
        else:
            # Method 2: Find any JSON object
            m = re.search(r'(\{[\s\S]*\})', raw)
            if m:
                json_str = m.group(1)
        
        if not json_str:
            logger.error(f"[judge_kg_relevance] Failed to parse JSON")
            logger.error(f"relevance raw response: {raw}")
            return {"relevant_kg": []}
        
        try:
            obj = json.loads(json_str)
            rel = obj.get("relevant_kg", [])
            if not isinstance(rel, list):
                rel = []
            rel = [r for r in rel if isinstance(r, str)]
            return {"relevant_kg": rel}
        except Exception as e:
            logger.error(f"[judge_kg_relevance] Failed to parse JSON: {e}")
            logger.error(f"relevance raw response: {raw}")
            logger.error(f"extracted json string: {json_str}")
            return {"relevant_kg": []}

class TaskLLM(curator.LLM):
    return_completions_object = True

    def prompt(self, input_sample):
        # Directly call original Task's prompt logic
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
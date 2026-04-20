import logging, sys
logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

import re
from bespokelabs import curator
from utils.parsing_utils import extract_answer, huatuo_match_choice

class Task:

    def get_user_prompt(self, item: dict) -> str:
        prompt = item['prompt']
        return f"""
{prompt}
Please remember to return your final answer choice within \\boxed{{}}.
"""

    def parse_output(self, item: dict) -> dict:
        response_text = item.get("raw_model_response", "")
            
        think_match = re.search(r'<think>(.*?)</think>', response_text, re.S)
        think_content = think_match.group(1).strip() if think_match else ""

        if not think_content:
            logging.warning(f"No think content found in response found.")

        return {
            "mcq_model_response": response_text,
            "think_content": think_content
        }


class TaskLLM(curator.LLM):
    return_completions_object = True

    def prompt(self, input_sample):
        return Task().get_user_prompt(input_sample)

    def parse(self, input_sample, response):
        message = response["choices"][0]["message"]
        #reasoning_content = message["reasoning_content"]
        reasoning_content = message.get("reasoning_content", "")
        content = message["content"]
        
        # logger.info(f"Reasoning content: {reasoning_content}")
        # logger.info(f"Model response content: {content}")

        if reasoning_content:
            raw = f"<think>\n{reasoning_content}\n</think>\n\n{content}"
        else:
            raw = content

        mid = {**input_sample, "raw_model_response": raw}
        new_fields = Task().parse_output(mid)

        result = {**input_sample, **new_fields}
        result.pop("raw_model_response", None)
        return result


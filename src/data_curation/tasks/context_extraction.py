import json, logging, re
from bespokelabs import curator
from utils.parsing_utils import extract_json_from_response

class Task:
    def get_user_prompt(self, item: dict) -> str:
        prompt = item.get("prompt", "").strip()
        return f"""Extract the background context from the following medical question. The context should include only factual statements or setup information, not the question itself. If there is no such context, return an empty string for "context".

Example:
Original Question:
"An otherwise healthy 30-year-old woman experiences intermittent headaches for 2 weeks. What is the likely diagnosis?"
Response:
{{"context": "An otherwise healthy 30-year-old woman experiences intermittent headaches for 2 weeks."}}

Now process the question below:
Original Question:
{prompt}

Return ONLY a JSON object with a single key "context", for example:
{{"context": "..."}}
"""

    def parse_output(self, item: dict) -> dict:
        raw = item.get("raw_model_response", "").strip()
        
        try:
            obj = extract_json_from_response(raw)
            if obj:
                ctx = obj.get("context", "").strip()
                return {"context": ctx}
        except Exception as e:
            logging.warning(f"context_extraction parse failed: {e}")
        
        return {"context": ""}

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


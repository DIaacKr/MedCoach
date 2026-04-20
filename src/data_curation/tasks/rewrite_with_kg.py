import json, logging, re
from bespokelabs import curator

from utils.parsing_utils import extract_json_from_response

logger = logging.getLogger(__name__)

class Task:
    """Rewrite grounded_text using retrieved KG triples."""

    def get_user_prompt(self, item: dict) -> str:
        subq    = item.get("subquestion", "")
        gt      = item.get("grounded_text", "")
        kg_list = item.get("relevant_kg", [])

        context = item.get("context", "")
        if context:
            context_block = f"Context:\n{context}\n\n"
        else:
            context_block = ""

        if kg_list:
            limited_kg_list = kg_list[:10]
            kg_block = "\n".join(
                f"- {kg}"
                for kg in limited_kg_list
            )
        else:
            logger.warning(f"[rewrite_with_kg] No relevant triples retrieved for subquestion: {subq}")
            kg_block = "(no relevant triples retrieved)"

        prompt_lines = [
            "You are a medical text rewriting assistant.",
            "Your task is to produce a clear, coherent answer that DIRECTLY addresses the given question by integrating relevant knowledge triples.",
            context_block,
            f"Question:\n{subq}",
            f"Original text:\n{gt}",
            "Knowledge triples (one per line):",
            kg_block,
            "",
            "Note: All relationships are undirected (e.g., 'parent-child' indicates a connection without specifying direction"
            "Please rewrite the original text into a complete, self-contained answer to the question.",
            "You may modify, expand, or reorganize the original text, and flexibly apply the knowledge triples if provided.",
            "When using the provided knowledge triples, only incorporate those that are truly relevant and helpful for answering the question. Ignore any knowledge that is not directly applicable to solving the problem at hand.",
            "If the original text contains factual errors or excessive uncertainty that would prevent a clear answer to the question, you may correct these issues and change the meaning as needed to better address the given question.",
            "If the given knowledge triples expression is slightly inaccurate or broad, you should apply the corresponding correct version"
            "Return ONLY a JSON object with the single field:",
            "{\"rewritten_text\": \"<your rewritten answer>\"}",
            "Do NOT include any additional explanation or mention retrieval."
        ]
        return "\n".join([line for line in prompt_lines if line])

    def parse_output(self, item: dict) -> dict:
        raw = item["raw_model_response"]
        original = item["grounded_text"]

        think_end = re.search(r'</think>', raw, re.IGNORECASE)
        content = raw[think_end.end():].strip() if think_end else raw

        try:
            obj = extract_json_from_response(content)
            if obj and "rewritten_text" in obj:
                rewritten = obj["rewritten_text"].strip()
                return {
                    "rewritten_text": rewritten,
                    "modified": rewritten.strip() != original.strip()
                }
        except Exception as e:
            logger.error(f"Failed to parse JSON in rewrite response: {e}")
        
        return {
            "rewritten_text": original,
            "modified": False
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

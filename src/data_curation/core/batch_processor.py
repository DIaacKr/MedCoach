import tqdm
from datasets import Dataset

class BatchProcessor:
    def __init__(self, client, tokenizer, batch_size, max_new_tokens, temperature=0.1, top_p=0.9):
        self.client = client
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

    def process_dataset(self, dataset: Dataset, task) -> Dataset:
        
        system_prompt = task.system_prompt if hasattr(task, 'system_prompt') else ""
        user_prompts = [task.get_user_prompt(item) for item in tqdm.tqdm(dataset, desc="Generating User Prompts")]

        all_prompts = []
        for p in user_prompts:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": p})

            formatted = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            all_prompts.append(formatted)

        responses = []
        for i in tqdm.tqdm(range(0, len(all_prompts), self.batch_size), desc="inference"):
            batch_prompts = all_prompts[i:i+self.batch_size]
            completion = self.client.completions.create(
                model="default",
                prompt=batch_prompts,
                max_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                #timeout=60, 
            )
            responses.extend([choice.text for choice in completion.choices])
        
        processed_dataset = dataset.add_column("raw_model_response", responses)

        final_dataset = processed_dataset.map(task.parse_output, desc="parsing model responses")
        
        final_dataset = final_dataset.remove_columns("raw_model_response")

        return final_dataset

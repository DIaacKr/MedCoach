import re
import dotenv
dotenv.load_dotenv()

import json
import datasets
import numpy as np
from functools import partial

def jprint(data):
    print(json.dumps(data, indent=2, ensure_ascii=False))

def load_medqa(seed=42, verbose=True):
    """
    https://huggingface.co/datasets/GBaker/MedQA-USMLE-4-options
    """
    name = "GBaker/MedQA-USMLE-4-options"
    split = "train"
    dataset = datasets.load_dataset(name, split=split)

    def _map_medqa(x, source_name, rng):
        raw_options = x["options"]
        question = x["question"]
        answer = x["answer_idx"]
        source = source_name

        answer_string = raw_options[answer]
        options = list(raw_options.values())
        rng.shuffle(options)
        answer_idx = options.index(answer_string)
        answer_letter = chr(ord("A") + answer_idx)
        letter_options = {chr(ord("A") + i): ans for i, ans in enumerate(options)}

        # NOTE: options format
        options = "\n".join([f"{op}. {ans}" for op, ans in letter_options.items()])

        # NOTE: prompt template, different datasets may vary
        prompt_template = "{question}\n{options}"
        prompt = prompt_template.format(question=question, options=options)

        return {
            # related
            # "question": question,
            # "answer": answer,
            # "options": raw_options,
            # source
            "source": source,
            "metadata": str(x),
            # llm inputs
            "options": letter_options,
            "prompt": prompt,
            "answer_letter": answer_letter,
            "answer_idx": answer_idx,
            "answer_string": answer_string,
        }

    if verbose:
        print("Dataset info:")
        jprint(dataset[0])

    rng = np.random.default_rng(seed=seed)
    source_name = f"{name}"
    mapped_sample = _map_medqa(dataset[0], source_name=source_name, rng=rng)

    if verbose:
        print("\nMapped sample:")
        jprint(mapped_sample)

    mapped_dataset: datasets.Dataset
    mapped_dataset = dataset.map(
        partial(_map_medqa, source_name=source_name, rng=rng),
        remove_columns=dataset.column_names,
    )

    return mapped_dataset

def load_medmcqa(seed=42, verbose=True):
    """
    https://huggingface.co/datasets/openlifescienceai/medmcqa
    """
    path = "openlifescienceai/medmcqa"
    split = "train"
    dataset = datasets.load_dataset(path, split=split)

    medmcqa_answer_mapping = dict(enumerate("ABCD"))

    def _map_medmcqa(x, source_name, answer_mapping, rng):
        question = x["question"]

        raw_options = {
            "A": x["opa"],
            "B": x["opb"],
            "C": x["opc"],
            "D": x["opd"],
        }
        raw_answer_idx = x["cop"]
        answer_letter = answer_mapping[raw_answer_idx]
        answer_string = raw_options[answer_letter]

        options = list(raw_options.values())
        rng.shuffle(options)
        answer_idx = options.index(answer_string)
        answer_letter = chr(ord("A") + answer_idx)
        letter_options = {chr(ord("A") + i): ans for i, ans in enumerate(options)}

        options = "\n".join([f"{op}. {ans}" for op, ans in letter_options.items()])

        # NOTE: prompt template, different datasets may vary
        prompt_template = "{question}\n{options}"
        prompt = prompt_template.format(question=question, options=options)

        return {
            # related
            # "question": question,
            # "raw_answer_idx": raw_answer_idx,
            # "options": raw_options,
            # source
            "source": source_name,
            "metadata": str(x),
            "options": letter_options,            
            # llm inputs
            "prompt": prompt,
            "answer_letter": answer_letter,
            "answer_idx": answer_idx,
            "answer_string": answer_string,
        }

    if verbose:
        print("Dataset info:")
        jprint(dataset[0])

    rng = np.random.default_rng(seed=seed)
    source_name = path
    mapped_sample = _map_medmcqa(
        dataset[0],
        source_name=source_name,
        answer_mapping=medmcqa_answer_mapping,
        rng=rng,
    )

    if verbose:
        print("\nMapped sample:")
        jprint(mapped_sample)

    mapped_dataset: datasets.Dataset
    mapped_dataset = dataset.map(
        partial(
            _map_medmcqa,
            source_name=source_name,
            rng=rng,
            answer_mapping=medmcqa_answer_mapping,
        ),
        remove_columns=dataset.column_names,
    )

    return mapped_dataset

def load_pubmedqa(seed=42, get_custom_test_split=False, verbose=True):
    """
    https://huggingface.co/datasets/qiaojin/PubMedQA

    Data split, 500 train, 500 test can be found in https://github.com/pubmedqa/pubmedqa/blob/master/preprocess/split_dataset.py

    Format is from https://github.com/FreedomIntelligence/HuatuoGPT-o1/blob/main/evaluation/data/eval_data.json
    """

    path = "qiaojin/PubMedQA"
    name = "pqa_labeled"
    split = "train"
    dataset = datasets.load_dataset(path, name=name, split=split)

    def _map_pubmedqa(x, source_name, rng):
        context = "\n".join(x["context"]["contexts"])
        question = x["question"]
        final_decision = x["final_decision"]
        long_answer = x["long_answer"]

        # NOTE: build options
        options = ["yes", "no", "maybe"]
        if final_decision not in options:
            raise ValueError(f"final_decision {final_decision} not in {options}")

        # NOTE: shuffle options to avoid memorization
        rng.shuffle(options)
        answer_idx = options.index(final_decision)
        answer_letter = chr(ord("A") + answer_idx)
        answer_string = final_decision

        letter_options = {chr(ord("A") + i): ans for i, ans in enumerate(options)}

        # NOTE: options format
        options = "\n".join([f"{op}. {ans}" for op, ans in letter_options.items()])

        # NOTE: prompt template, different datasets may vary
        prompt_template = "{context}\n{question}\n{options}"
        prompt = prompt_template.format(
            context=context, question=question, options=options
        )

        return {
            # related
            # "context": context,
            # "question": question,
            # "final_decision": final_decision,
            # "long_answer": long_answer,
            # source
            "source": source_name,
            "metadata": str(x),
            # llm inputs
            "options": letter_options,
            "prompt": prompt,
            "answer_letter": answer_letter,
            "answer_idx": answer_idx,
            "answer_string": answer_string,
        }

    if verbose:
        print("Dataset info:")
        jprint(dataset[0])

    rng = np.random.default_rng(seed=seed)
    source_name = f"{path}:{name}"
    mapped_sample = _map_pubmedqa(dataset[0], source_name=source_name, rng=rng)
    
    if verbose:
        print("\nMapped sample:")
        jprint(mapped_sample)

    mapped_dataset: datasets.Dataset
    mapped_dataset = dataset.map(
        partial(_map_pubmedqa, source_name=source_name, rng=rng),
        remove_columns=dataset.column_names,
    )

    # NOTE: split dataset, 500: 500
    # https://huggingface.co/docs/datasets/v3.3.2/en/process#split
    # seed: 0 https://github.com/pubmedqa/pubmedqa/blob/master/preprocess/split_dataset.py
    PUBMEDQA_SEED = 0
    mapped_dataset = mapped_dataset.train_test_split(
        test_size=0.5, seed=PUBMEDQA_SEED, shuffle=True
    )   

    if get_custom_test_split:
        mapped_dataset = mapped_dataset["test"]
    else:
        mapped_dataset = mapped_dataset["train"]
    return mapped_dataset

def load_headqa(seed=42, verbose=True):
    """
    https://huggingface.co/datasets/openlifescienceai/headqa
    """
    name = "openlifescienceai/headqa"
    split = "train"
    dataset = datasets.load_dataset(name, split=split)

    def _map_headqa(x, source_name, rng):
        """
        {
        "Correct Answer": "Internal mitochondrial",
        "Correct Option": "A",
        "Options": {
        "A": "Internal mitochondrial",
        "B": "External mitochondrial",
        "C": "Plasma.",
        "D": "Lysosomal"
        },
        "Question": "The cardiolipin phospholipid is abundant in the membrane:"
        }
        """
        data = x["data"]

        question = data["Question"]
        raw_answer_string = data["Correct Answer"]
        raw_answer_idx = data["Correct Option"]
        raw_options = data["Options"]
        if raw_answer_string != raw_options[raw_answer_idx]:
            raise ValueError(
                f"Answer not in options, `{raw_answer_string}` not in `{raw_options}`"
            )

        options = list(raw_options.values())
        rng.shuffle(options)
        answer_idx = options.index(raw_answer_string)
        answer_letter = chr(ord("A") + answer_idx)
        letter_options = {chr(ord("A") + i): ans for i, ans in enumerate(options)}
        answer_string = raw_answer_string

        # NOTE: options format
        options = "\n".join([f"{op}. {ans}" for op, ans in letter_options.items()])

        # NOTE: prompt template, different datasets may vary
        prompt_template = "{question}\n{options}"
        prompt = prompt_template.format(question=question, options=options)
        return {
            # related
            # "question": question,
            # "raw_answer_string": raw_answer_string,
            # "raw_answer_idx": raw_answer_idx,
            # "options": raw_options,
            # source
            "source": source_name,
            "metadata": str(x),
            # llm inputs
            "options": letter_options,
            "prompt": prompt,
            "answer_letter": answer_letter,
            "answer_idx": answer_idx,
            "answer_string": answer_string,
        }

    if verbose:
        print("Dataset info:")
        jprint(dataset[0])

    rng = np.random.default_rng(seed=seed)
    source_name = f"{name}"
    mapped_sample = _map_headqa(dataset[0], source_name=source_name, rng=rng)

    if verbose:
        print("\nMapped sample:")
        jprint(mapped_sample)

    mapped_dataset: datasets.Dataset
    mapped_dataset = dataset.map(
        partial(_map_headqa, source_name=source_name, rng=rng),
        remove_columns=dataset.column_names,
    )

    return mapped_dataset

def load_m1k_tokenized_self(seed=42, verbose=True):
    """
    https://huggingface.co/datasets/UCSC-VLAA/m1k-tokenized
    """
    name = "UCSC-VLAA/m1k-tokenized"
    split = "train"
    dataset = datasets.load_dataset(name, split=split)

    def _map(x, source_name, rng):
        question = x["prompt"]
        reasoning = x["reasoning"]
        response = x["distilled_answer_string"]

        answer_letter = x["answer_letter"]
        answer_string = x["answer_string"]
        answer_idx = x["answer_idx"]

        lines = question.strip().split('\n')
        option_lines = [line for line in lines if re.match(r"^[A-Z]\.", line)]

        if not option_lines:
            raise ValueError(f"No options found in question: {question}")
        
        letter_options = {}
        for line in option_lines:
            parts = line.split('.', 1)
            if len(parts) == 2:
                letter = parts[0].strip()
                text = parts[1].strip()
                letter_options[letter] = text

        wrong_options = {k: v for k, v in letter_options.items() if k != answer_letter}
        if not wrong_options:
            raise ValueError(f"No wrong options found for answer letter: {answer_letter}")
        else:
            wrong_answer_letter = rng.choice(list(wrong_options.keys()))
            wrong_answer_string = wrong_options[wrong_answer_letter]

        return {
            "source": source_name,
            "metadata": str(x),
            #"question": question,
            #"reasoning": reasoning,
            #"response": response,
            #"text": text,
            "prompt": question,
            "options": letter_options,
            "answer_letter": answer_letter,
            "answer_string": answer_string,
            "answer_idx": answer_idx,
            "wrong_answer_letter": wrong_answer_letter,
            "wrong_answer_string": wrong_answer_string,
        }

    if verbose:
        print("Dataset info:")
        jprint(dataset[0])

    rng = np.random.default_rng(seed=seed)
    source_name = f"{name}"
    mapped_sample = _map(dataset[0], source_name=source_name, rng=rng)

    if verbose:
        print("\nMapped sample:")
        if mapped_sample:
            jprint(mapped_sample)
        else:
            print("None")

    mapped_dataset: datasets.Dataset
    mapped_dataset = dataset.map(
        partial(_map, source_name=source_name, rng=rng),
        remove_columns=dataset.column_names,
    )

    mapped_dataset = mapped_dataset.filter(lambda x: x is not None)

    if verbose:
        print(f"Mapped dataset: {len(mapped_dataset)} samples")

    return mapped_dataset

dataset_loaders = {
    "medqa": load_medqa,
    "pubmedqa": load_pubmedqa,
    "headqa": load_headqa,
    "medmcqa": load_medmcqa,
    "m1kself": load_m1k_tokenized_self,
}

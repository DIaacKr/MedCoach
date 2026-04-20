import multiprocessing
import os
import pickle
import sys
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import argparse
import logging
import torch
import gc
import importlib
import openai
from tqdm import tqdm
from transformers import AutoTokenizer
from datasets import Dataset
from torch.utils.data import DataLoader

from core.sglang_server import SGLangServer
from core.batch_processor import BatchProcessor

from triplet2text_dict import relation_templates


logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(handler)
logger.setLevel(logging.INFO)
logger.propagate = False


def triple_to_text(relation: str,
                   display_relation: str,
                   x_type: str,
                   y_type: str,
                   x_name: str,
                   y_name: str) -> str:
    type1, type2 = sorted([x_type, y_type])
    key = (relation, display_relation, type1, type2)

    if key not in relation_templates:
        rev_key = (relation, display_relation, type2, type1)
        if rev_key not in relation_templates:
            raise KeyError(f"No template for key {key} or reversed key {rev_key}")
        key = rev_key
        x_type, y_type = y_type, x_type
        x_name, y_name = y_name, x_name
    
    template = relation_templates[key]

    if key[2] == x_type:
        subj_name, obj_name = x_name, y_name
    else:
        subj_name, obj_name = y_name, x_name

    return template.format(subject=subj_name, object=obj_name)

def main():
    parser = argparse.ArgumentParser(description="Use LLMs to convert KG triplets into sentences, which are then embedded and indexed.")
    parser.add_argument("--kg_csv_path", default='./data/kg.csv')
    parser.add_argument("--output_index_path", default='./data/kg_index.faiss')
    parser.add_argument("--output_meta_path", default='./data/kg_metadata.parquet')
    parser.add_argument("--kg_meta_map_path", default='./data/kg_meta_map.pkl')
    parser.add_argument("--entity_pool_path", default='./data/entity_pool.pkl')
    parser.add_argument("--embed_model", default='abhinand/MedEmbed-large-v0.1')
    parser.add_argument("--embed_batch_size", type=int, default=128)
    parser.add_argument("--num_samples", type=int, default=None)

    
    args = parser.parse_args()

    if os.path.exists(args.output_meta_path):
        df = pd.read_parquet(args.output_meta_path, engine='pyarrow')
    else:
        df = pd.read_csv(args.kg_csv_path, dtype=str)
        if args.num_samples and args.num_samples > 0:
            df = df.head(args.num_samples)
        logger.info(f"all keys in csv: {df.columns.tolist()}")

        sentences = []
        for row in tqdm(df.itertuples(index=False), total=len(df), desc="Generating sentences"):
            sentences.append(triple_to_text(
                row.relation, row.display_relation,
                row.x_type, row.y_type,
                row.x_name, row.y_name
            ))
        df["kg_sentence"] = sentences

        df.to_parquet(
            args.output_meta_path,
            engine='pyarrow',
            compression='snappy',
            index=False
        )

        kg_meta_map = {
            row["kg_sentence"]: row
            for row in df.to_dict(orient="records")
        }
        entity_pool = {}
        for t in set(df.x_type) | set(df.y_type):
            names = (
                df.loc[df.x_type == t, 'x_name'].tolist() +
                df.loc[df.y_type == t, 'y_name'].tolist()
            )
            entity_pool[t] = list(set(names))

        
        os.makedirs(os.path.dirname(args.kg_meta_map_path), exist_ok=True)
        os.makedirs(os.path.dirname(args.entity_pool_path), exist_ok=True)
        with open(args.kg_meta_map_path, "wb") as f:
            pickle.dump(kg_meta_map, f)
        with open(args.entity_pool_path, "wb") as f:
            pickle.dump(entity_pool, f)

    device = "cuda"
    embedder = SentenceTransformer(args.embed_model, device=device)

    embeddings = embedder.encode(
        df["kg_sentence"].tolist(),
        batch_size=args.embed_batch_size,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    faiss.normalize_L2(embeddings)
    dim = embeddings.shape[1]
    index = faiss.IndexIDMap(faiss.IndexFlatIP(dim))
    ids = np.arange(len(embeddings), dtype='int64')
    index.add_with_ids(embeddings, ids)
    faiss.write_index(index, args.output_index_path)

    
    logger.info(f"Finished, saved in {args.output_index_path} and {args.output_meta_path}")

if __name__ == '__main__':
    main()

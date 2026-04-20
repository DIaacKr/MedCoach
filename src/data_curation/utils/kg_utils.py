import os, json, logging
import pandas as pd
import faiss
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from datasets import Dataset
import numpy as np

logger = logging.getLogger(__name__)

def retrieve_topk_kg(
    input_dataset,
    output_path: str,
    kg_index: str,
    kg_meta: str,
    embed_model: str,
    top_k: int = 5,
    batch_size: int = 128  
):

    index = faiss.read_index(kg_index)
    meta  = pd.read_parquet(kg_meta, engine="pyarrow")
    assert len(meta) == index.ntotal

    embedder = SentenceTransformer(embed_model, device='cuda')

    total = len(input_dataset)
    
    with open(output_path, "w", encoding="utf-8") as writer:
        for i in tqdm(range(0, total, batch_size), desc="Retrieving KG batches"):
            batch_indices = list(range(i, min(i + batch_size, total)))
            batch_dataset = input_dataset.select(batch_indices)
            
            batch_items = batch_dataset.to_list() if hasattr(batch_dataset, 'to_list') else [batch_dataset[j] for j in range(len(batch_dataset))]
            
            queries = [f"Q: {item['subquestion']}\n A: {item['grounded_text']}" 
                      for item in batch_items]
            
            embs = embedder.encode(queries, convert_to_numpy=True, show_progress_bar=False)
            faiss.normalize_L2(embs)
            
            D, I = index.search(embs, top_k)
            
            for j, item in enumerate(batch_items):
                topk = []
                for idx, score in zip(I[j], D[j]):
                    kg = meta.iloc[idx].to_dict()
                    kg["score"] = float(score)
                    topk.append(kg)
                item["top_kg"] = topk
                writer.write(json.dumps(item, ensure_ascii=False) + "\n")

    logger.info(f"[KG] Retrieval completed for {total} records, results written to {output_path}")
    
    records = []
    with open(output_path, "r", encoding="utf-8") as _f:
        for _line in _f:
            _line = _line.strip()
            if not _line:
                continue
            records.append(json.loads(_line))
    ds2 = Dataset.from_list(records)
    return ds2

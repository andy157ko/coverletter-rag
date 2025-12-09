# scripts/build_embeddings.py
import json
from pathlib import Path
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

PROC_DIR = Path("data/processed")
EMB_DIR = Path("data/embeddings")

def main():
    embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    records = []
    with (PROC_DIR / "train.jsonl").open() as f:
        for line in f:
            records.append(json.loads(line))

    texts = [
        r["job_text"] 
        + "\n" 
        + r["resume_text"] 
        + "\n\nCOVER LETTER:\n" 
        + r["cover_letter_text"]
        for r in records
    ]
    vectors = embed_model.encode(texts, convert_to_numpy=True).astype("float32")

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    EMB_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(EMB_DIR / "faiss_index.bin"))

    #Save metadata to look up examples at retrieval time
    with (EMB_DIR / "metadata.jsonl").open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

if __name__ == "__main__":
    main()

import faiss
import numpy as np
import os
from pathlib import Path

def build_faiss_index(embedding_path: str, output_path: str):
    # Create FAISS index based on cosine similarity
    """Tạo FAISS index dựa trên cosine similarity."""
    Path(os.path.dirname(output_path)).mkdir(parents=True, exist_ok=True)
    vecs = np.load(embedding_path).astype("float32")
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)
    faiss.write_index(index, output_path)
    print(f"FAISS index built → {output_path}")
    return index

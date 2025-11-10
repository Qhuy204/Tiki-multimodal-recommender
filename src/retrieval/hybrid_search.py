import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

def search_similar_text(query: str, products_path: str, embedding_path: str, index_path: str, top_k: int = 5):
    # Search for similar products using text query
    df = pd.read_csv(products_path)
    model = SentenceTransformer("keepitreal/vietnamese-sbert")
    q_vec = model.encode([query]).astype("float32")
    index = faiss.read_index(index_path)
    sims, ids = index.search(q_vec, top_k)
    results = df.iloc[ids[0]][["product_id", "name", "price", "brand"]].assign(score=sims[0])
    print("Top-K Search Results:")
    print(results)
    return results

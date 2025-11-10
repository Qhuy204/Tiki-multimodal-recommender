import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# Paths
INDEX_PATH = "models/faiss_index/tiki.index"
PRODUCTS_CSV = "Data/processed/normalized/products.csv"
TEXT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Load index and data
index = faiss.read_index(INDEX_PATH)
df = pd.read_csv(PRODUCTS_CSV)
text_encoder = SentenceTransformer(TEXT_MODEL)

print(f"[INFO] Loaded FAISS index dimension: {index.d}")

while True:
    query = input("\nNhập truy vấn tìm sản phẩm (hoặc 'q' để thoát): ").strip()
    if query.lower() == "q":
        break

    # Encode text query
    q_text = text_encoder.encode([query]).astype("float32")
    print(f"[DEBUG] q_text shape: {q_text.shape}")

    # If FAISS index is multimodal (1280D), we need to pad image zeros
    if index.d == 1280 and q_text.shape[1] == 768:
        q_img = np.zeros((1, 512), dtype=np.float32)
        q_emb = np.concatenate([q_text, q_img], axis=1)
    else:
        q_emb = q_text

    print(f"[DEBUG] q_emb shape: {q_emb.shape}, index expects {index.d}")

    # Validate dimension match
    if q_emb.shape[1] != index.d:
        print(f"[ERROR] Query dim {q_emb.shape[1]} ≠ index dim {index.d}")
        continue

    # Perform search
    D, I = index.search(q_emb, k=5)

    print("\nKết quả gần nhất:")
    for rank, idx in enumerate(I[0]):
        if idx < len(df):
            row = df.iloc[idx]
            print(f"{rank+1}. {row['name']} | {row['brand']} | score={D[0][rank]:.4f}")

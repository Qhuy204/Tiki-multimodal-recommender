import sys, os, time, traceback
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Data_Processor.data_extractor import extract_tiki_data
from Data_Processor.data_cleaner import clean_text_fields
from src.data.preprocess_text import generate_text_embeddings
from src.data.preprocess_image import generate_image_embeddings
from src.data.feature_engineering import merge_embeddings
from src.retrieval.index_builder import build_faiss_index
import pandas as pd
from pathlib import Path
import numpy as np

# Path configuration
RAW_PATH = "Data/raw/tiki_dataset.jsonl"
PRODUCTS_CSV = "Data/processed/normalized/products.csv"
TEXT_CLEAN_CSV = "Data/processed/normalized/text_clean.csv"
TEXT_EMB = "Data/embeddings/text_embeddings.npy"
IMG_EMB = "Data/embeddings/image_embeddings.npy"
MULTI_EMB = "Data/embeddings/multimodal_embeddings.npy"
FAISS_INDEX = "models/faiss_index/tiki.index"
IMAGE_DIR = "/home/qhuy/DA3/Data/raw/images"

# Run a single pipeline step with timing and error handling
def run_step(step_name, func, *args, **kwargs):
    print(f"\n[{step_name}] Starting...")
    t0 = time.time()
    try:
        func(*args, **kwargs)
        print(f"[{step_name}] Done in {time.time() - t0:.2f}s")
    except Exception as e:
        print(f"[{step_name}] Failed: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    start = time.time()
    print("Starting Tiki ETL + Embedding + Index pipeline...")

    run_step("1. Extracting data", extract_tiki_data, RAW_PATH, PRODUCTS_CSV)
    run_step("2. Cleaning text fields", clean_text_fields, PRODUCTS_CSV, TEXT_CLEAN_CSV)
    run_step("3. Generating text embeddings", generate_text_embeddings, TEXT_CLEAN_CSV, TEXT_EMB)

    # Image embeddings stage
    if Path(IMAGE_DIR).exists() and len(os.listdir(IMAGE_DIR)) > 0:
        run_step(
            "4. Generating image embeddings",
            generate_image_embeddings,
            PRODUCTS_CSV,
            IMG_EMB,
            IMAGE_DIR,
            64,
            4
        )
    else:
        print(f"[4] Skipping image embeddings (folder not found or empty): {IMAGE_DIR}")
        np.save(IMG_EMB, np.empty((0, 512)))

    run_step("5. Merging multimodal embeddings", merge_embeddings, TEXT_EMB, IMG_EMB, MULTI_EMB)
    run_step("6. Building FAISS index", build_faiss_index, MULTI_EMB, FAISS_INDEX)

    print(f"\nPipeline completed successfully in {time.time() - start:.2f}s")

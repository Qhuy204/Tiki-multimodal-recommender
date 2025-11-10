import json
import pandas as pd
from pathlib import Path

def extract_tiki_data(input_path: str, output_path: str):
    """
    Extract clean product metadata and user-item interactions from Tiki JSONL data.
    - Removes duplicate products by product_id
    - Saves:
        1. products.csv: product info
        2. user_item_interactions.csv: (user_id, product_id, timestamp)
    """
    products, interactions = [], []
    input_path, output_path = Path(input_path), Path(output_path)

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                item = json.loads(line)
                detail = item.get("product_detail", {})
                pid = item.get("product_id") or detail.get("id")
                if not pid:
                    continue

                # Product fields
                products.append({
                    "product_id": pid,
                    "category": item.get("category"),
                    "name": detail.get("name"),
                    "price": detail.get("price"),
                    "discount_rate": detail.get("discount_rate"),
                    "brand": (detail.get("brand") or {}).get("name"),
                    "rating": detail.get("rating_average"),
                    "sold": (detail.get("quantity_sold") or {}).get("value"),
                    "short_description": detail.get("short_description"),
                    "description": detail.get("description"),
                })

                # User-item interactions from reviews
                reviews = item.get("reviews", {}).get("data", [])
                for r in reviews:
                    user_id = r.get("customer_id") or r.get("created_by", {}).get("id")
                    if not user_id:
                        continue
                    ts = r.get("timeline", {}).get("review_created_date")
                    interactions.append({
                        "user_id": user_id,
                        "product_id": pid,
                        "timestamp": ts
                    })

            except Exception:
                continue

    # DataFrames
    df_products = pd.DataFrame(products)
    df_interactions = pd.DataFrame(interactions)

    # Deduplicate
    df_products.drop_duplicates(subset=["product_id"], inplace=True)
    df_interactions.drop_duplicates(subset=["user_id", "product_id"], inplace=True)

    # Save outputs
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_products.to_csv(output_path, index=False)
    inter_path = output_path.parent / "user_item_interactions.csv"
    df_interactions.to_csv(inter_path, index=False)

    print(f"- Extracted {len(df_products)} unique products → {output_path}")
    print(f"- Extracted {len(df_interactions)} user-item pairs → {inter_path}")

    return df_products, df_interactions

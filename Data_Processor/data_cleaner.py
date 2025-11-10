import pandas as pd
import re
from bs4 import BeautifulSoup
from pathlib import Path
from src.utils.gemini_client import clean_with_gemini

def clean_html(text: str) -> str:
    """
    Remove HTML tags, <img>, links, and irrelevant sections.
    Keep only the main product description content.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # Parse HTML
    soup = BeautifulSoup(text, "html.parser")

    # Remove unwanted tags: images, links, scripts, etc.
    for tag in soup(["img", "a", "script", "style", "iframe", "noscript"]):
        tag.decompose()

    # Extract text
    text = soup.get_text(separator=" ")

    # Remove FAQ, related search, or footer sections
    patterns_to_remove = [
        r"Câu hỏi thường gặp.*",                # FAQs
        r"Một số nội dung tìm kiếm.*",          # Related search
        r"Giá sản phẩm trên Tiki.*",            # Footer
        r"HSD.*ngày SX.*",                      # Expiry/manufacture
    ]
    for pat in patterns_to_remove:
        text = re.sub(pat, "", text, flags=re.IGNORECASE | re.DOTALL)

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", " ", text)

    # Remove special characters and repeated punctuation
    text = re.sub(r"[^\w\sÀ-ỹà-ỹ.,!?%€₫]", " ", text)
    text = re.sub(r"(\*{1,}|-+|•+|–+|…+|_+)", " ", text)
    text = re.sub(r"([.!?]){2,}", r"\1", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def clean_text_fields(input_path: str, output_path: str):
    """
    Clean and combine text fields into 'text_full'.
    Rewrites the original 'description' with cleaned text.
    """
    df = pd.read_csv(input_path)

    # Clean HTML from description
    # df["description"] = df["description"].apply(clean_html)
    # Lấy toàn bộ mô tả thành list
    descriptions = df["description"].astype(str).tolist()

    # Làm sạch theo batch
    cleaned_texts = clean_with_gemini(descriptions)

    # Gán kết quả trở lại DataFrame
    df["description"] = cleaned_texts


    # Combine text fields for embedding
    df["text_full"] = (
        df["name"].fillna("") + ". " +
        df["short_description"].fillna("") + ". " +
        df["description"].fillna("")
    ).str.strip()

    # Drop empty text rows
    df = df[df["text_full"].str.len() > 0]

    # Save results
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Cleaned and normalized text saved → {output_path}")
    return df

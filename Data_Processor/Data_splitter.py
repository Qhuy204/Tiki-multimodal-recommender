import json
import pandas as pd
from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning  # Để clean HTML và ignore warning
import warnings
import re
import os
from typing import Dict, List, Any

# Ignore BeautifulSoup warning về URL-like input
warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)

def clean_text(text: str) -> str:
    """
    Làm sạch text: remove HTML tags, extra whitespace, special characters.
    Giữ nguyên tiếng Việt (không lowercase để giữ nguyên ý nghĩa cho NLP).
    """
    if pd.isna(text) or text is None:
        return ""
    
    # Remove HTML tags using BeautifulSoup
    soup = BeautifulSoup(str(text), 'html.parser')
    text = soup.get_text(separator=' ', strip=True)
    
    # Normalize whitespace: replace multiple spaces/newlines with single space
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove special chars but keep Vietnamese accents (e.g., á, â, etc.)
    # Chỉ remove non-alphanumeric except Vietnamese chars
    text = re.sub(r'[^\w\s\u00C0-\u1EF9]', ' ', text)  # \u00C0-\u1EF9 covers Vietnamese
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def extract_specifications(specs: List[Dict]) -> Dict[str, str]:
    """
    Trích xuất specifications thành dict {key: value}.
    Key là 'name' của attribute (e.g., "Thương hiệu"), value là 'value'.
    """
    spec_dict = {}
    if not specs:
        return spec_dict
    for group in specs:
        for attr in group.get('attributes', []):
            key = attr.get('name', '').strip()
            value = attr.get('value', '').strip()
            if key and value:
                spec_dict[key] = value
    return spec_dict

def extract_reviews(reviews: Dict) -> Dict[str, Any]:
    """
    Trích xuất reviews: 
    - review_texts: list of cleaned text
    - review_ratings: list of stars (parallel to texts)
    - stars_distribution: dict {1: count, 2: count, ...}
    - rating_average: float
    - review_count: int
    Chỉ lấy reviews có text.
    """
    result = {
        'review_texts': [],
        'review_ratings': [],
        'stars_distribution': {},
        'rating_average': reviews.get('rating_average', 0.0),
        'review_count': reviews.get('reviews_count', 0)
    }
    
    stars = reviews.get('stars', {})
    for star_key in stars:
        count = stars.get(star_key, {}).get('count', 0)
        if count > 0:
            result['stars_distribution'][int(star_key)] = count
    
    data = reviews.get('data', [])
    for review in data:
        text = clean_text(review.get('content', ''))
        rating = review.get('rating', 0)
        if text and len(text) > 10:  # Filter short meaningless reviews
            result['review_texts'].append(text)
            result['review_ratings'].append(rating)
    
    return result

def extract_breadcrumbs(breadcrumbs: List[Dict]) -> List[str]:
    """
    Trích xuất breadcrumbs thành list category names (hierarchy).
    """
    names = []
    for crumb in breadcrumbs:
        name = crumb.get('name', '').strip()
        if name:
            names.append(name)
    return names

def extract_images(images: List[Dict]) -> List[str]:
    """
    Trích xuất URLs của images (base_url, ưu tiên first 5 để multimodal).
    Filter valid URLs.
    """
    urls = []
    for img in images or []:
        base_url = img.get('base_url', '')
        if base_url and base_url.startswith('http') and 'tikicdn.com' in base_url:
            urls.append(base_url)
    return urls[:5]  # Limit to top 5 for efficiency in CV

def process_product_record(record: Dict) -> Dict[str, Any]:
    """
    Xử lý một record JSON thành dict chuẩn hóa.
    Bao quát tất cả fields cần thiết cho các components.
    """
    detail = record.get('product_detail', {})
    reviews = record.get('reviews', {})
    specs = detail.get('specifications', [])
    
    # Common fields
    product_id = detail.get('id', 0)
    name_clean = clean_text(detail.get('name', ''))
    category = record.get('category', detail.get('categories', {}).get('name', ''))
    brand = detail.get('brand', {}).get('name', '')
    description_clean = clean_text(detail.get('description', ''))
    short_desc_clean = clean_text(detail.get('short_description', ''))
    specs_dict = extract_specifications(specs)
    breadcrumbs_list = extract_breadcrumbs(detail.get('breadcrumbs', []))
    images_list = extract_images(detail.get('images', []))
    reviews_extracted = extract_reviews(reviews)
    
    # 1. Product Recommender System
    recommender_data = {
        'product_id': product_id,
        'name': name_clean,
        'category': category,
        'brand': brand,
        'price': detail.get('price', 0),
        'original_price': detail.get('original_price', 0),
        'rating_average': detail.get('rating_average', reviews.get('rating_average', 0.0)),
        'review_count': detail.get('review_count', reviews.get('reviews_count', 0)),
        'favourite_count': detail.get('favourite_count', 0),
        'specifications': specs_dict,  # Dict
        'all_time_quantity_sold': detail.get('all_time_quantity_sold', 0),
        'day_ago_created': detail.get('day_ago_created', 0),
        'breadcrumbs': breadcrumbs_list  # List[str]
    }
    
    # 2. Sentiment Filtering
    sentiment_data = {
        'product_id': product_id,
        'review_texts': reviews_extracted['review_texts'],
        'review_ratings': reviews_extracted['review_ratings'],
        'stars_distribution': reviews_extracted['stars_distribution'],
        'rating_average': reviews_extracted['rating_average'],
        'review_count': reviews_extracted['review_count'],
        'description': description_clean  # Mô tả sản phẩm cho sentiment nếu cần
    }
    
    # 3. Multimodal Search (CV + NLP)
    multimodal_data = {
        'product_id': product_id,
        'images': images_list,
        'name': name_clean,  # Reuse
        'description': description_clean,  # Reuse
        'short_description': short_desc_clean,
        'specifications': specs_dict,  # Reuse
        'categories': category,  # Simple category
        'breadcrumbs': breadcrumbs_list  # Hierarchy
    }
    
    # 4. RAG (Retrieval-Augmented Generation)
    # Concat all text for RAG context
    top_reviews_text = ' '.join(reviews_extracted['review_texts'][:3]) if reviews_extracted['review_texts'] else ''
    all_text = ' '.join([
        name_clean,
        description_clean,
        category,
        ' '.join([f"{k}: {v}" for k, v in specs_dict.items()]),
        top_reviews_text
    ]).strip()
    
    rag_data = {
        'product_id': product_id,
        'rag_context_text': all_text,
        'structured_data': {  # JSON-like for RAG prompt
            'price': recommender_data['price'],
            'rating': recommender_data['rating_average'],
            'category': category,
            'brand': brand
        }
    }
    
    return {
        'recommender': recommender_data,
        'sentiment': sentiment_data,
        'multimodal': multimodal_data,
        'rag': rag_data
    }

def save_to_csv(df: pd.DataFrame, output_path: str, list_cols: List[str] = None, dict_cols: List[str] = None):
    """
    Helper: Save DF to CSV, json.dumps lists/dicts với ensure_ascii=False để giữ ký tự tiếng Việt.
    """
    if list_cols:
        for col in list_cols:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, list) else '[]')
    if dict_cols:
        for col in dict_cols:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, dict) else '{}')
    
    # Sử dụng utf-8-sig để hỗ trợ Excel mở file với ký tự đúng
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"Đã lưu: {output_path} ({len(df)} rows)")

def standardize_data(input_file: str, root_folder: str = '/data') -> Dict[str, pd.DataFrame]:
    """
    Chuẩn hóa toàn bộ data từ .jsonl file.
    - Chia thành 4 files riêng biệt cho từng component trong subfolder normalized.
    - Mỗi file có 'product_id' làm key để dễ join sau (e.g., pd.merge on product_id).
    - Root folder: /data (sẽ tạo /data/normalized).
    - Handle large file line-by-line.
    
    Args:
        input_file: Path to .jsonl (e.g., 'D:\\VSCode\\DA3\\tiki_dataset_clean.jsonl')
        root_folder: Root folder (e.g., '/data' hoặc 'D:\\data').
    Returns:
        Dict { 'recommender': df, 'sentiment': df, ... } cho further use.
    """
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"File not found: {input_file}")
    
    # Ensure root folder và subdir
    normalized_dir = os.path.join(root_folder, 'normalized')
    os.makedirs(normalized_dir, exist_ok=True)
    
    processed_data = {
        'recommender': [],
        'sentiment': [],
        'multimodal': [],
        'rag': []
    }
    error_count = 0
    
    print("Đang đọc và chuẩn hóa data từ .jsonl...")
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                record = json.loads(line.strip())
                norm_dict = process_product_record(record)
                
                # Append to each component list
                for comp, data in norm_dict.items():
                    processed_data[comp].append(data)
                
            except json.JSONDecodeError as e:
                print(f"Line {line_num}: JSON decode error - {e}")
                error_count += 1
            except Exception as e:
                print(f"Record {line_num}: Processing error - {e}")
                error_count += 1
            
            if line_num % 10000 == 0:
                print(f"Processed {line_num} records...")
    
    # Create DFs and save
    dfs = {}
    for comp in processed_data:
        df_comp = pd.DataFrame(processed_data[comp])
        output_path = os.path.join(normalized_dir, f'{comp}.csv')
        
        # Specific list/dict cols per component
        if comp == 'recommender':
            list_cols = ['breadcrumbs']
            dict_cols = ['specifications']
        elif comp == 'sentiment':
            list_cols = ['review_texts', 'review_ratings']
            dict_cols = ['stars_distribution']
        elif comp == 'multimodal':
            list_cols = ['images', 'breadcrumbs']
            dict_cols = ['specifications']
        else:  # rag
            list_cols = []
            dict_cols = ['structured_data']
        
        save_to_csv(df_comp, output_path, list_cols, dict_cols)
        dfs[comp] = df_comp
    
    print(f"\nHoàn thành! Chuẩn hóa {len(processed_data['recommender'])} records (errors: {error_count}).")
    print(f"Files lưu tại: {normalized_dir}/")
    print("- recommender.csv")
    print("- sentiment.csv")
    print("- multimodal.csv")
    print("- rag.csv")
    
    # Stats chi tiết cho từng component
    print("\n=== STATS PRODUCT RECOMMENDER ===")
    rec_cols = ['product_id', 'name', 'category', 'brand', 'price', 'original_price',
                'rating_average', 'review_count', 'favourite_count', 'all_time_quantity_sold', 'day_ago_created']
    print(dfs['recommender'][rec_cols].describe(include='all'))
    
    print("\n=== STATS SENTIMENT FILTERING ===")
    sent_cols = ['product_id', 'review_count', 'rating_average', 'description']
    print(dfs['sentiment'][sent_cols].describe(include='all'))
    if 'review_texts' in dfs['sentiment'].columns:
        print(f"Avg reviews per product: {dfs['sentiment']['review_texts'].apply(lambda x: len(json.loads(x)) if x != '[]' else 0).mean():.2f}")
    
    print("\n=== STATS MULTIMODAL SEARCH ===")
    multi_cols = ['product_id', 'name', 'description', 'short_description', 'categories']
    print(dfs['multimodal'][multi_cols].describe(include='all'))
    if 'images' in dfs['multimodal'].columns:
        print(f"Avg images per product: {dfs['multimodal']['images'].apply(lambda x: len(json.loads(x)) if x != '[]' else 0).mean():.2f}")
    
    print("\n=== STATS RAG ===")
    rag_cols = ['product_id', 'rag_context_text']
    print(dfs['rag'][rag_cols].describe(include='all'))
    print(f"Avg RAG text length: {dfs['rag']['rag_context_text'].str.len().mean():.2f} chars")
    
    # Sample 3 rows per file (sử dụng display để tránh escape trong print)
    print("\n=== SAMPLE 3 ROWS PER FILE ===")
    for comp in dfs:
        print(f"\n--- {comp.upper()} ---")
        print(dfs[comp].head(3)[['product_id', 'name', 'category']].to_string())  # Giới hạn columns để dễ đọc
        # Để xem full, dùng pd.read_csv và json.loads sau
    
    return dfs

# Chạy code (thay path nếu cần; root_folder='/data' sẽ tạo /data/normalized)
if __name__ == "__main__":
    input_path = r'D:\VSCode\DA3\tiki_dataset_clean.jsonl'
    root_folder = r'D:\VSCode\DA3\data'
    dfs = standardize_data(input_path, root_folder)
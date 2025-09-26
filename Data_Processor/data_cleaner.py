import pandas as pd
import numpy as np
import json
import re
from typing import Dict, List, Any, Optional
from urllib.parse import urlparse
import warnings
warnings.filterwarnings('ignore')

class TikiDataPreprocessor:
    def __init__(self):
        self.processed_data = {}
        
    def load_data(self, file_path: str) -> List[Dict]:
        """Load JSON data from file (supports both .json and .jsonl formats)"""
        data = []
        
        if file_path.endswith('.jsonl'):
            # Handle JSONL format (one JSON object per line)
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f):
                    line = line.strip()
                    if line:  # Skip empty lines
                        try:
                            data.append(json.loads(line))
                        except json.JSONDecodeError as e:
                            print(f"Error parsing line {line_num + 1}: {e}")
                            continue
        else:
            # Handle regular JSON format
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        
        return data
    
    def extract_product_features(self, products: List[Dict]) -> pd.DataFrame:
        """Extract core product features for recommendation system"""
        
        product_features = []
        
        for item in products:
            try:
                product = item.get('product_detail', {})
                
                # Basic product info
                features = {
                    'product_id': product.get('id'),
                    'master_id': product.get('master_id'),
                    'name': product.get('name', ''),
                    'short_description': product.get('short_description', ''),
                    'url_key': product.get('url_key', ''),
                    'type': product.get('type', ''),
                    
                    # Pricing
                    'price': product.get('price', 0),
                    'list_price': product.get('list_price', 0),
                    'original_price': product.get('original_price', 0),
                    'discount': product.get('discount', 0),
                    'discount_rate': product.get('discount_rate', 0),
                    
                    # Ratings & Reviews
                    'rating_average': product.get('rating_average', 0),
                    'review_count': product.get('review_count', 0),
                    'favourite_count': product.get('favourite_count', 0),
                    
                    # Sales metrics
                    'all_time_quantity_sold': product.get('all_time_quantity_sold', 0),
                    'day_ago_created': product.get('day_ago_created', 0),
                    
                    # Brand & Origin - Safe access
                    'brand_name': product.get('brand', {}).get('name', '') if product.get('brand') else '',
                    'brand_id': product.get('brand', {}).get('id', '') if product.get('brand') else '',
                    
                    # Category - Safe access
                    'category_name': product.get('categories', {}).get('name', '') if product.get('categories') else '',
                    'category_id': product.get('categories', {}).get('id', '') if product.get('categories') else '',
                    
                    # Inventory
                    'inventory_status': product.get('inventory_status', ''),
                    'inventory_type': product.get('inventory_type', ''),
                    
                    # Additional flags
                    'is_fresh': product.get('is_fresh', False),
                    'is_flower': product.get('is_flower', False),
                    'has_buynow': product.get('has_buynow', False),
                    'is_gift_card': product.get('is_gift_card', False),
                }
                
                # Extract specifications - Safe access
                specs = product.get('specifications', []) or []
                for spec_group in specs:
                    if spec_group and isinstance(spec_group, dict):
                        for attr in spec_group.get('attributes', []) or []:
                            if attr and isinstance(attr, dict):
                                key = f"spec_{attr.get('code', '')}"
                                features[key] = attr.get('value', '')
                
                # Extract category hierarchy - Safe access
                breadcrumbs = product.get('breadcrumbs', []) or []
                for i, crumb in enumerate(breadcrumbs[:-1]):  # Exclude last item (product itself)
                    if crumb and isinstance(crumb, dict):
                        features[f'category_level_{i+1}_name'] = crumb.get('name', '')
                        features[f'category_level_{i+1}_id'] = crumb.get('category_id', '')
                
                product_features.append(features)
                
            except Exception as e:
                print(f"Error processing product {item.get('product_detail', {}).get('id', 'unknown')}: {e}")
                continue
        
        return pd.DataFrame(product_features)
    
    def extract_image_features(self, products: List[Dict]) -> pd.DataFrame:
        """Extract image URLs and metadata for computer vision tasks"""
        
        image_data = []
        
        for item in products:
            product = item.get('product_detail', {})
            product_id = product.get('id')
            
            images = product.get('images', [])
            for idx, img in enumerate(images):
                image_info = {
                    'product_id': product_id,
                    'image_index': idx,
                    'base_url': img.get('base_url', ''),
                    'large_url': img.get('large_url', ''),
                    'medium_url': img.get('medium_url', ''),
                    'small_url': img.get('small_url', ''),
                    'thumbnail_url': img.get('thumbnail_url', ''),
                    'is_gallery': img.get('is_gallery', False),
                    'label': img.get('label', ''),
                }
                image_data.append(image_info)
        
        return pd.DataFrame(image_data)
    
    def extract_text_features(self, products: List[Dict]) -> pd.DataFrame:
        """Extract text content for NLP tasks and RAG"""
        
        text_data = []
        
        for item in products:
            product = item.get('product_detail', {})
            
            # Clean HTML from description
            description = product.get('description', '')
            description_clean = re.sub(r'<[^>]+>', '', description)
            description_clean = re.sub(r'\s+', ' ', description_clean).strip()
            
            text_info = {
                'product_id': product.get('id'),
                'name': product.get('name', ''),
                'short_description': product.get('short_description', ''),
                'description': description_clean,
                'description_raw': description,
                
                # Meta information
                'meta_title': product.get('meta_title', ''),
                'meta_description': product.get('meta_description', ''),
                'meta_keywords': product.get('meta_keywords', ''),
                
                # Combine all text for full-text search
                'combined_text': ' '.join([
                    product.get('name', ''),
                    product.get('short_description', ''),
                    description_clean,
                    product.get('brand', {}).get('name', ''),
                    product.get('categories', {}).get('name', ''),
                ]).strip(),
                
                # Category text
                'category_path': ' > '.join([
                    crumb.get('name', '') for crumb in product.get('breadcrumbs', [])[:-1]
                ]),
                
                # Brand
                'brand_name': product.get('brand', {}).get('name', ''),
            }
            
            # Add specifications as text
            specs_text = []
            for spec_group in product.get('specifications', []):
                for attr in spec_group.get('attributes', []):
                    specs_text.append(f"{attr.get('name', '')}: {attr.get('value', '')}")
            text_info['specifications_text'] = ' | '.join(specs_text)
            
            text_data.append(text_info)
        
        return pd.DataFrame(text_data)
    
    def extract_review_features(self, products: List[Dict]) -> pd.DataFrame:
        """Extract review data for sentiment analysis"""
        
        review_data = []
        
        for item in products:
            try:
                product_detail = item.get('product_detail', {})
                reviews = item.get('reviews', {})
                top_reviews = item.get('top_reviews', [])
                
                product_id = product_detail.get('id')
                
                # Skip if no product_id
                if not product_id:
                    continue
                
                # Overall review statistics
                review_stats = {
                    'product_id': product_id,
                    'rating_average': reviews.get('rating_average', 0) if reviews else 0,
                    'reviews_count': reviews.get('reviews_count', 0) if reviews else 0,
                    'total_photos': reviews.get('review_photo', {}).get('total_photo', 0) if reviews and reviews.get('review_photo') else 0,
                    
                    # Star distribution - Safe access
                    'stars_1': reviews.get('stars', {}).get('1', {}).get('count', 0) if reviews and reviews.get('stars') and reviews.get('stars').get('1') else 0,
                    'stars_2': reviews.get('stars', {}).get('2', {}).get('count', 0) if reviews and reviews.get('stars') and reviews.get('stars').get('2') else 0,
                    'stars_3': reviews.get('stars', {}).get('3', {}).get('count', 0) if reviews and reviews.get('stars') and reviews.get('stars').get('3') else 0,
                    'stars_4': reviews.get('stars', {}).get('4', {}).get('count', 0) if reviews and reviews.get('stars') and reviews.get('stars').get('4') else 0,
                    'stars_5': reviews.get('stars', {}).get('5', {}).get('count', 0) if reviews and reviews.get('stars') and reviews.get('stars').get('5') else 0,
                    
                    'stars_1_percent': reviews.get('stars', {}).get('1', {}).get('percent', 0) if reviews and reviews.get('stars') and reviews.get('stars').get('1') else 0,
                    'stars_2_percent': reviews.get('stars', {}).get('2', {}).get('percent', 0) if reviews and reviews.get('stars') and reviews.get('stars').get('2') else 0,
                    'stars_3_percent': reviews.get('stars', {}).get('3', {}).get('percent', 0) if reviews and reviews.get('stars') and reviews.get('stars').get('3') else 0,
                    'stars_4_percent': reviews.get('stars', {}).get('4', {}).get('percent', 0) if reviews and reviews.get('stars') and reviews.get('stars').get('4') else 0,
                    'stars_5_percent': reviews.get('stars', {}).get('5', {}).get('percent', 0) if reviews and reviews.get('stars') and reviews.get('stars').get('5') else 0,
                }
                
                # Calculate sentiment metrics
                total_reviews = sum([
                    review_stats['stars_1'], review_stats['stars_2'], review_stats['stars_3'],
                    review_stats['stars_4'], review_stats['stars_5']
                ])
                
                if total_reviews > 0:
                    review_stats['positive_ratio'] = (review_stats['stars_4'] + review_stats['stars_5']) / total_reviews
                    review_stats['negative_ratio'] = (review_stats['stars_1'] + review_stats['stars_2']) / total_reviews
                    review_stats['neutral_ratio'] = review_stats['stars_3'] / total_reviews
                else:
                    review_stats['positive_ratio'] = 0
                    review_stats['negative_ratio'] = 0
                    review_stats['neutral_ratio'] = 0
                
                review_data.append(review_stats)
                
                # Individual reviews (if available)
                if reviews and reviews.get('data'):
                    for review in reviews.get('data', []):
                        if not review:  # Skip None reviews
                            continue
                            
                        # Safe access to created_by field
                        created_by = review.get('created_by') or {}
                        
                        individual_review = {
                            'product_id': product_id,
                            'review_id': review.get('id'),
                            'rating': review.get('rating', 0),
                            'title': review.get('title', ''),
                            'content': review.get('content', ''),
                            'created_at': review.get('created_at', ''),
                            'customer_name': created_by.get('name', '') if isinstance(created_by, dict) else '',
                            'is_purchased': created_by.get('purchased', False) if isinstance(created_by, dict) else False,
                            'helpful_count': review.get('thank_count', 0),
                            'has_image': len(review.get('images', [])) > 0 if review.get('images') else False,
                        }
                        review_data.append(individual_review)
                        
            except Exception as e:
                print(f"Error processing reviews for product {item.get('product_detail', {}).get('id', 'unknown')}: {e}")
                continue
        
        return pd.DataFrame(review_data)
    
    def create_user_item_matrix(self, df_products: pd.DataFrame) -> pd.DataFrame:
        """Create user-item interaction matrix for collaborative filtering"""
        
        # For now, we'll use implicit feedback based on sales, ratings, and favorites
        user_item_data = []
        
        for _, product in df_products.iterrows():
            # Simulate user interactions based on product popularity
            num_interactions = max(1, int(product['all_time_quantity_sold'] + product['favourite_count']))
            
            for user_id in range(1, min(num_interactions + 1, 100)):  # Limit to 100 simulated users per product
                interaction = {
                    'user_id': f"user_{user_id}",
                    'product_id': product['product_id'],
                    'rating': max(1, min(5, np.random.normal(product['rating_average'], 0.5))),
                    'interaction_type': 'implicit',
                    'timestamp': pd.Timestamp.now() - pd.Timedelta(days=np.random.randint(0, 365))
                }
                user_item_data.append(interaction)
        
        return pd.DataFrame(user_item_data)
    
    def calculate_content_features(self, df_products: pd.DataFrame) -> pd.DataFrame:
        """Calculate additional content-based features"""
        
        df_features = df_products.copy()
        
        # Price features
        df_features['price_log'] = np.log1p(df_features['price'])
        df_features['discount_amount'] = df_features['list_price'] - df_features['price']
        df_features['is_on_sale'] = df_features['discount_rate'] > 0
        df_features['price_tier'] = pd.qcut(df_features['price'], q=5, labels=['very_low', 'low', 'medium', 'high', 'very_high'], duplicates='drop')
        
        # Popularity features
        df_features['popularity_score'] = (
            df_features['all_time_quantity_sold'] * 0.4 +
            df_features['favourite_count'] * 0.3 +
            df_features['review_count'] * 0.2 +
            df_features['rating_average'] * df_features['review_count'] * 0.1
        )
        
        # Freshness features
        df_features['is_new'] = df_features['day_ago_created'] <= 30
        df_features['age_category'] = pd.cut(
            df_features['day_ago_created'], 
            bins=[0, 30, 90, 365, float('inf')], 
            labels=['new', 'recent', 'old', 'very_old']
        )
        
        # Quality indicators
        df_features['quality_score'] = np.where(
            df_features['review_count'] > 0,
            df_features['rating_average'] * np.log1p(df_features['review_count']),
            0
        )
        
        return df_features
    
    def process_all_data(self, file_path: str) -> Dict[str, pd.DataFrame]:
        """Main function to process all data"""
        
        print("Loading data...")
        products = self.load_data(file_path)
        print(f"Loaded {len(products)} products")
        
        print("Extracting product features...")
        df_products = self.extract_product_features(products)
        
        print("Extracting image features...")
        df_images = self.extract_image_features(products)
        
        print("Extracting text features...")
        df_text = self.extract_text_features(products)
        
        print("Extracting review features...")
        df_reviews = self.extract_review_features(products)
        
        print("Creating user-item matrix...")
        df_interactions = self.create_user_item_matrix(df_products)
        
        print("Calculating content features...")
        df_content_features = self.calculate_content_features(df_products)
        
        self.processed_data = {
            'products': df_content_features,
            'images': df_images,
            'text': df_text,
            'reviews': df_reviews,
            'interactions': df_interactions
        }
        
        return self.processed_data
    
    def save_processed_data(self, output_dir: str):
        """Save processed data to CSV files with UTF-8 BOM for Excel compatibility"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        for name, df in self.processed_data.items():
            output_path = os.path.join(output_dir, f"{name}.csv")
            
            # Save with UTF-8 BOM (best for Excel, keeps Vietnamese accents)
            df.to_csv(output_path, index=False, encoding='utf-8-sig')
            print(f"Saved {name} data to {output_path} ({len(df)} records)")

                
    def save_with_multiple_encodings(self, output_dir: str):
        """Save data with multiple encoding options"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        encodings = {
            'utf8': 'utf-8',
            'utf8_bom': 'utf-8-sig',  # Best for Excel
            'excel': 'cp1252'         # Windows Excel encoding
        }
        
        for name, df in self.processed_data.items():
            for enc_name, encoding in encodings.items():
                try:
                    output_path = os.path.join(output_dir, f"{name}_{enc_name}.csv")
                    
                    if encoding == 'cp1252':
                        # Replace characters that can't be encoded
                        df.to_csv(output_path, index=False, encoding=encoding, errors='replace')
                    else:
                        df.to_csv(output_path, index=False, encoding=encoding)
                        
                    print(f"Saved {name} ({encoding}) to {output_path}")
                except Exception as e:
                    print(f"Failed to save {name} with {encoding}: {e}")
    
    def get_data_summary(self):
        """Print summary of processed data"""
        for name, df in self.processed_data.items():
            print(f"\n{name.upper()} DATA:")
            print(f"Shape: {df.shape}")
            print(f"Columns: {list(df.columns)[:10]}...")  # Show first 10 columns
            if len(df) > 0:
                print(f"Sample data:")
                print(df.head(2))

# Usage example
if __name__ == "__main__":
    # Initialize preprocessor
    preprocessor = TikiDataPreprocessor()
    
    # Process all data
    processed_data = preprocessor.process_all_data(r'D:\VSCode\DA3\tiki_dataset_clean.jsonl')
    
    # Get summary
    preprocessor.get_data_summary()
    
    # Save processed data
    preprocessor.save_processed_data('processed_data')
    
    print("\nData preprocessing completed!")
    print("\nFiles generated:")
    print("- products.csv: Core product features for recommendation")
    print("- images.csv: Image URLs for computer vision")  
    print("- text.csv: Text content for NLP and RAG")
    print("- reviews.csv: Review data for sentiment analysis")
    print("- interactions.csv: User-item interactions for collaborative filtering")
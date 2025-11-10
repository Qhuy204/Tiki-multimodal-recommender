import pandas as pd
import numpy as np
import re
import faiss
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import plotly.express as px
from transformers import AutoTokenizer, AutoModel
import torch
import warnings
warnings.filterwarnings('ignore')

# =====================
# Load data
# =====================
@st.cache_data
def load_data(file_path='../products_features_cleaned.csv'):
    df = pd.read_csv(file_path)
    return df

df = load_data()

# =====================
# Preprocessing
# =====================
def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    return text.strip()

df['text_combined'] = (
    df['name'].apply(clean_text) + ' ' +
    df['description'].apply(clean_text).fillna('') + ' ' +
    df['specifications'].apply(clean_text).fillna('')
)

# Store original price
df['price_original'] = df['price'].copy()

# Normalize numeric features
scaler = StandardScaler()
num_features = ['price', 'rating_avg']
df[num_features] = scaler.fit_transform(df[num_features].fillna(0))

# Category encoding
def parse_taxonomy(breadcrumbs):
    if pd.isna(breadcrumbs):
        return ['']
    return breadcrumbs.split(' > ')

df['taxonomy_levels'] = df['breadcrumbs'].apply(parse_taxonomy)
df['top_category'] = df['taxonomy_levels'].apply(lambda x: x[0] if x else '')

le = LabelEncoder()
df['cat_encoded'] = le.fit_transform(df['top_category'])

# =====================
# Embeddings + FAISS
# =====================
@st.cache_resource
def build_indexes():
    # Load PhoBERT
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
    model = AutoModel.from_pretrained("vinai/phobert-base")

    # Hàm encode với PhoBERT
    def phobert_encode(texts, batch_size=32):
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            enc = tokenizer(batch, padding=True, truncation=True,
                            max_length=256, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**enc)
                # Mean pooling
                last_hidden = outputs.last_hidden_state
                mask = enc["attention_mask"].unsqueeze(-1).expand(last_hidden.size()).float()
                summed = torch.sum(last_hidden * mask, 1)
                counts = torch.clamp(mask.sum(1), min=1e-9)
                mean_pooled = summed / counts
            embeddings.append(mean_pooled.cpu().numpy())
        return np.vstack(embeddings).astype("float32")

    # Encode text
    text_embeddings = phobert_encode(df['text_combined'].tolist(), batch_size=32)

    # Metadata vector (numeric + category)
    meta_features = np.hstack([
        df[num_features].values,
        df['cat_encoded'].values.reshape(-1,1)
    ]).astype('float32')

    # Build FAISS index
    d_text = text_embeddings.shape[1]
    index_text = faiss.IndexFlatIP(d_text)
    faiss.normalize_L2(text_embeddings)
    index_text.add(text_embeddings)

    return tokenizer, model, text_embeddings, meta_features, index_text

model, text_embeddings, meta_features, index_text = build_indexes()

# =====================
# Hybrid recommendation
# =====================
def hybrid_search(query, k=10, alpha=0.8, beta=0.2):
    q_emb = phobert_encode([clean_text(query)])
    faiss.normalize_L2(q_emb)

    D, I = index_text.search(q_emb, k*5)
    candidates = I[0]
    sim_text = D[0]

    q_meta = np.zeros((1, meta_features.shape[1]), dtype=np.float32)
    sim_meta = cosine_similarity(q_meta, meta_features[candidates])[0]

    final_score = alpha * sim_text + beta * sim_meta
    top_idx = candidates[np.argsort(final_score)[::-1][:k]]

    return df.iloc[top_idx][['product_id', 'name', 'price_original', 'top_category']]


# =====================
# Streamlit UI
# =====================
st.title("Hybrid Product Recommendation Demo")

query = st.text_input("Nhập tên sản phẩm (ví dụ: 'kem chống nắng')")
if query:
    results = hybrid_search(query, k=10)
    st.subheader("Top 10 gợi ý")
    st.dataframe(results)

# Sidebar filters
st.sidebar.header("Filters")
price_max = float(df['price_original'].max())
price_filter = st.sidebar.slider("Max Price", 0.0, price_max, price_max, step=1.0)
cat_filter = st.sidebar.multiselect("Categories", df['top_category'].unique())

filtered_df = df[df['price_original'] <= price_filter]
if cat_filter:
    filtered_df = filtered_df[filtered_df['top_category'].isin(cat_filter)]

st.subheader("Filtered Products")
st.dataframe(filtered_df[['product_id', 'name', 'price_original']])

col1, col2 = st.columns(2)
with col1:
    fig1 = px.histogram(df, x='price_original', title="Price Distribution")
    st.plotly_chart(fig1)
with col2:
    fig2 = px.pie(df, names='top_category', title="Category Coverage")
    st.plotly_chart(fig2)

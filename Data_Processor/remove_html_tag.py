import pandas as pd
import re

# Đọc file CSV mà user đã upload
file_path = "products_features.csv"
df = pd.read_csv(file_path)

# Hàm loại bỏ các thẻ HTML và các thẻ tương tự
def remove_html_tags(text):
    if pd.isna(text):  # Nếu giá trị là NaN thì giữ nguyên
        return text
    clean = re.compile(r'<.*?>')
    return re.sub(clean, '', str(text))

# Áp dụng cho toàn bộ DataFrame
df_cleaned = df.applymap(remove_html_tags)

# Lưu file mới
output_path = "products_features_cleaned.csv"
df_cleaned.to_csv(output_path, index=False)

output_path

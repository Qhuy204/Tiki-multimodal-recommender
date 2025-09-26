import pandas as pd

# Đọc file CSV gốc
input_file = r'D:\VSCode\DA3\products_features.csv'
output_file = "100_sampled.csv"  

# Đọc dữ liệu
df = pd.read_csv(input_file)

# Lấy ngẫu nhiên 100 dòng (nếu file có >100 dòng)
sampled_df = df.sample(n=100, random_state=42)

# Lưu ra file CSV mới
sampled_df.to_csv(output_file, index=False)

print(f"Đã lưu 100 samples vào {output_file}")

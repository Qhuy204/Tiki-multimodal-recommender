import os
import json
import requests
from PIL import Image
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed

# Config
input_file = r"D:\VSCode\DA3\tiki_dataset_clean.jsonl"   # file JSONL đầu vào
output_dir = r"E:\DA3\Data\images"  # thư mục lưu ảnh resize
target_size = (224, 224)
max_workers = 16  # số luồng (có thể tăng lên 16 nếu mạng khỏe)

os.makedirs(output_dir, exist_ok=True)

def download_and_resize(product_id, url, idx, size=(224,224)):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content)).convert("RGB")
        img = img.resize(size, Image.LANCZOS)
        filename = os.path.join(output_dir, f"{product_id}_{idx}.jpg")
        img.save(filename, format="JPEG", quality=85)
        return filename
    except Exception as e:
        return f"❌ Failed {url}: {e}"

# Đọc JSONL
tasks = []
with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        record = json.loads(line)
        product_id = record["product_id"]
        images = record["product_detail"].get("images", [])
        for i, img in enumerate(images):
            url = img.get("base_url")
            if url:
                tasks.append((product_id, url, i))

print(f"📸 Tổng số ảnh cần tải: {len(tasks)}")

# Tải ảnh song song
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = [executor.submit(download_and_resize, pid, url, idx, target_size) 
               for pid, url, idx in tasks]

    for i, future in enumerate(as_completed(futures), 1):
        result = future.result()
        if i % 500 == 0:
            print(f"✅ Đã xử lý {i}/{len(tasks)} ảnh")

print("🎉 Hoàn tất download + resize ảnh!")

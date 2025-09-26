import requests
import csv
import time
import os

BASE_HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json"
}

def get_products_by_category(cat_id, max_pages=5, writer=None, buffer=None, chunk_size=10):
    """Crawl sản phẩm từ 1 category ID, lưu theo từng chunk_size"""
    for page in range(1, max_pages + 1):
        url = (
            f"https://tiki.vn/api/personalish/v1/blocks/listings"
            f"?limit=40&include=advertisement&aggregations=2"
            f"&category={cat_id}&page={page}"
        )
        r = requests.get(url, headers=BASE_HEADERS)
        if r.status_code != 200:
            print(f"[WARN] Lỗi {r.status_code} với category {cat_id}, page {page}")
            break

        data = r.json()
        items = data.get("data", [])
        if not items:  # hết dữ liệu
            break

        for item in items:
            product = {
                "id": item.get("id"),
                "name": item.get("name"),
                "price": item.get("price"),
                "discount": item.get("discount"),
                "rating_average": item.get("rating_average"),
                "review_count": item.get("review_count"),
                "category_id": cat_id,
                "url": f"https://tiki.vn/{item.get('url_path')}"
            }
            buffer.append(product)

            # đủ 10 bản ghi thì flush xuống file
            if len(buffer) >= chunk_size:
                writer.writerows(buffer)
                print(f"💾 Đã lưu {len(buffer)} bản ghi vào file")
                buffer.clear()

        time.sleep(0.5)  # tránh bị chặn

def crawl_from_csv(input_csv=r"D:\VSCode\DA3\tiki_leaf_categories.csv", output_csv="tiki_products.csv"):
    fieldnames = ["id", "name", "price", "discount", "rating_average", "review_count", "category_id", "url"]

    # nếu file chưa tồn tại thì ghi header trước
    write_header = not os.path.exists(output_csv)

    with open(output_csv, "a", newline="", encoding="utf-8") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        buffer = []  # bộ nhớ tạm để ghi theo chu kỳ

        with open(input_csv, newline="", encoding="utf-8") as f_in:
            reader = csv.DictReader(f_in)
            for row in reader:
                cat_id = row["id"]
                cat_name = row["name"]
                print(f"==> Crawling category {cat_name} ({cat_id})")

                get_products_by_category(cat_id, max_pages=3, writer=writer, buffer=buffer, chunk_size=10)

        # ghi nốt phần còn lại <10 bản ghi
        if buffer:
            writer.writerows(buffer)
            print(f"💾 Đã lưu {len(buffer)} bản ghi cuối cùng vào file")
            buffer.clear()

    print(f"✅ Crawl hoàn tất, dữ liệu đã lưu vào {output_csv}")

if __name__ == "__main__":
    crawl_from_csv()

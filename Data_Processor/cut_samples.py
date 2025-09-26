import json
import random

def cut_random_records(input_file, output_file, num_samples):
    """
    Cắt ngẫu nhiên một số bản ghi từ file JSONL.

    Args:
        input_file (str): Đường dẫn file gốc (.jsonl).
        output_file (str): Đường dẫn file mới để lưu kết quả.
        num_samples (int): Số bản ghi cần lấy ngẫu nhiên.
    """
    # Đọc toàn bộ bản ghi trong file jsonl
    with open(input_file, "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f if line.strip()]

    # Chọn ngẫu nhiên num_samples bản ghi
    sampled_records = random.sample(lines, min(num_samples, len(lines)))

    # Ghi ra file mới ở định dạng jsonl
    with open(output_file, "w", encoding="utf-8") as f:
        for record in sampled_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Đã cắt {len(sampled_records)} bản ghi và lưu vào {output_file}")


# Ví dụ sử dụng
if __name__ == "__main__":
    input_path = r"D:\VSCode\DA3\tiki_dataset.jsonl"       # file jsonl gốc
    output_path = "sampled.jsonl"   # file jsonl mới
    num_samples = 100               # số bản ghi muốn lấy ngẫu nhiên
    cut_random_records(input_path, output_path, num_samples)

from dotenv import load_dotenv
import os
import google.generativeai as genai
import time
import math
from concurrent.futures import ThreadPoolExecutor, as_completed

load_dotenv()

MAX_BATCH_SIZE = 20
MAX_WORKERS = 5
RETRY_DELAY = 2
MAX_RETRIES = 3

def clean_with_gemini(raw_html_list, api_key: str = None, retries: int = MAX_RETRIES):
    """
    Batch-clean product descriptions using Gemini 2.0 Flash.
    """
    key = api_key or os.getenv("GEMINI_API_KEY")
    if not key:
        raise ValueError("Missing GEMINI_API_KEY")
    genai.configure(api_key=key)
    model = genai.GenerativeModel("gemini-2.0-flash")

    def _process_batch(batch_texts):
        joined_text = "\n\n---\n\n".join(batch_texts)
        prompt = (
            "Bạn là một công cụ làm sạch dữ liệu mô tả sản phẩm tiếng Việt.\n\n"
            "Nhiệm vụ của bạn là làm sạch và chuẩn hóa các đoạn mô tả sau:\n"
            "- Loại bỏ toàn bộ thẻ HTML, thẻ <img>, <a>, đường dẫn URL, markdown, hoặc ký tự đặc biệt.\n"
            "- Giữ nguyên nội dung và ý nghĩa gốc, không được tóm tắt, không dịch, không thêm nội dung.\n"
            "- Gộp các câu lại thành các đoạn văn mạch lạc, tiếng Việt tự nhiên.\n"
            "- Chỉ xuất văn bản thuần, không thêm ký hiệu, đánh số hay tiêu đề. Xóa các dòng trống, ghép các đoạn rời rạc lại thành 1 đoạn thống nhất (mỗi sample là 1 đoạn, không được xuống dòng)\n\n"
            "Các mô tả cần làm sạch được phân tách bằng dòng chỉ chứa `---`.\n"
            f"{joined_text}\n\n"
            "Trả về đúng số lượng đoạn mô tả đầu vào, theo thứ tự, phân tách bằng dòng `---`."
        )

        for attempt in range(retries):
            try:
                response = model.generate_content(prompt)
                text = (response.text or "").strip()
                cleaned = [s.strip() for s in text.split("---") if s.strip()]
                if len(cleaned) < len(batch_texts):
                    cleaned += ["[Gemini output missing]"] * (len(batch_texts) - len(cleaned))
                return cleaned

            except Exception as e:
                if attempt < retries - 1:
                    time.sleep(RETRY_DELAY)
                    continue
                print(f"Gemini batch failed: {e}")
                return ["[Gemini error]"] * len(batch_texts)

    all_results = []
    batches = math.ceil(len(raw_html_list) / MAX_BATCH_SIZE)
    print(f"Total {len(raw_html_list)} samples → {batches} batches (~{MAX_BATCH_SIZE}/batch)")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []
        for i in range(0, len(raw_html_list), MAX_BATCH_SIZE):
            batch = raw_html_list[i:i + MAX_BATCH_SIZE]
            futures.append(executor.submit(_process_batch, batch))

        for f in as_completed(futures):
            all_results.extend(f.result())

    return all_results[:len(raw_html_list)]

import os
import random
import shutil

def copy_random_images(input_folder, output_folder, num_images=50):
    # Tạo thư mục output nếu chưa có
    os.makedirs(output_folder, exist_ok=True)

    # Lấy danh sách tất cả file ảnh trong input
    valid_exts = (".jpg", ".jpeg", ".png", ".bmp", ".gif")
    all_images = [f for f in os.listdir(input_folder) if f.lower().endswith(valid_exts)]

    if not all_images:
        print("❌ Không tìm thấy ảnh trong thư mục input.")
        return

    # Nếu ít ảnh hơn số cần chọn thì lấy hết
    selected = random.sample(all_images, min(num_images, len(all_images)))

    # Copy ảnh sang output
    for img in selected:
        src = os.path.join(input_folder, img)
        dst = os.path.join(output_folder, img)
        shutil.copy(src, dst)

    print(f"✅ Đã copy {len(selected)} ảnh từ {input_folder} sang {output_folder}")

# Ví dụ dùng
copy_random_images(r"D:\Dataset_Car\Car\images", r"D:\Dataset_Car\test", 50)

from PIL import Image
import cv2
import sys
import os

# Đảm bảo in ra tiếng Việt không bị lỗi
sys.stdout.reconfigure(encoding='utf-8')

def check_image_requirements(image_folder, image_filename):
    # Tạo đường dẫn ảnh
    image_path = os.path.join(image_folder, image_filename)

    # Kiểm tra file tồn tại
    if not os.path.exists(image_path):
        print(f"Không tìm thấy ảnh: {image_path}")
        return

    # Kiểm tra kích thước và chế độ màu
    img = Image.open(image_path)
    width, height = img.size
    mode = img.mode

    print(f"\n--- Kiểm tra ảnh: {image_filename} ---")
    print(f"Kích thước ảnh: {width} x {height}")
    print(f"Chế độ màu (PIL): {mode}")

    # Dùng OpenCV để kiểm tra số kênh màu
    img_cv = cv2.imread(image_path)
    if img_cv is None:
        print("Không thể đọc ảnh bằng OpenCV.")
        return

    channels = img_cv.shape[2] if len(img_cv.shape) == 3 else 1
    print(f"Số kênh màu (OpenCV): {channels}")

    # Kiểm tra điều kiện cho SSIM
    if width < 7 or height < 7:
        print("Ảnh quá nhỏ để tính SSIM (yêu cầu ít nhất 7x7 pixel)")
    else:
        print("Ảnh đủ điều kiện để tính SSIM")

    if mode != 'RGB':
        print("Ảnh không phải RGB")
    else:
        print("Ảnh là RGB")

# Ví dụ sử dụng
input_folder = 'output_images'
input_filename = 'watermarked.png'

check_image_requirements(input_folder, input_filename)

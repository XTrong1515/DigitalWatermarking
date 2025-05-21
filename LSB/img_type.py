from PIL import Image

def check_image_mode(image_path):
    """
    Kiểm tra loại ảnh (mode) của một file ảnh.
    Ví dụ: RGB, Grayscale (L), RGBA, CMYK, v.v...
    """
    try:
        img = Image.open(image_path)
        print(f"Đường dẫn: {image_path}")
        print(f"Loại ảnh (mode): {img.mode}")
        print(f"Kích thước ảnh: {img.size}")
        
        if img.mode == 'L':
            print("⇒ Đây là ảnh grayscale (đen trắng).")
        elif img.mode == 'RGB':
            print("⇒ Đây là ảnh màu RGB.")
        elif img.mode == 'RGBA':
            print("⇒ Đây là ảnh RGB có thêm kênh alpha (độ trong suốt).")
        elif img.mode == 'CMYK':
            print("⇒ Đây là ảnh màu CMYK (dùng trong in ấn).")
        else:
            print("⇒ Đây là ảnh có định dạng đặc biệt hoặc không phổ biến.")
    except Exception as e:
        print(f"[Lỗi] Không thể mở ảnh: {e}")

check_image_mode(r"C:\HocTap\BMMMT\digital_watermark\lsb\watermark_img\1000_F_539348176_ulGRbIS9rDObiEfl4MFrbwKNXQCe6SZC.jpg")

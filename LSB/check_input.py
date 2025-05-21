from PIL import Image

# Hàm kiểm tra xem ảnh watermark có phù hợp để nhúng vào ảnh gốc hay không
def is_watermark_compatible(image_path, watermark_path):
    # Mở ảnh gốc và ảnh watermark
    img = Image.open(image_path)
    watermark = Image.open(watermark_path)

    # Kiểm tra kích thước của ảnh watermark so với ảnh gốc
    img_width, img_height = img.size
    watermark_width, watermark_height = watermark.size

    # In kích thước của ảnh gốc và watermark
    print(f"Kích thước ảnh gốc: {img_width}x{img_height}")
    print(f"Kích thước ảnh watermark: {watermark_width}x{watermark_height}")

    if watermark_width > img_width or watermark_height > img_height:
        print(f"Ảnh watermark quá lớn so với ảnh gốc!")
        return False

    # Kiểm tra xem ảnh watermark có phải là ảnh grayscale (1 kênh màu) hay không
    if watermark.mode != 'L' and watermark.mode != 'RGB':
        print("Ảnh watermark không phải là ảnh grayscale hoặc RGB. Vui lòng sử dụng ảnh có chế độ 'L' hoặc 'RGB'.")
        return False

    # Nếu tất cả các điều kiện đều thỏa mãn
    print("Ảnh watermark phù hợp để nhúng vào ảnh gốc.")
    return True

# image_path = r'C:\HocTap\BMMMT\digital_watermark\lsb\input_images\IMG_9785.png'  # Đường dẫn đến ảnh gốc
# watermark_path = r'C:\HocTap\BMMMT\digital_watermark\lsb\watermark_img\extracted_watermark_img.jpg'  # Đường dẫn đến ảnh watermark

image_path = r'C:\HocTap\BMMMT\digital_watermark\lsb\input_images\stock-photo-159533631-1500x1000.jpg'  # Đường dẫn đến ảnh gốc
watermark_path = r'C:\HocTap\BMMMT\digital_watermark\lsb\watermark_img\miles.jpg'  # Đường dẫn đến ảnh watermark

# Kiểm tra tính tương thích của watermark
is_watermark_compatible(image_path, watermark_path)

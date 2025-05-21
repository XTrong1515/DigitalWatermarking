from PIL import Image
import numpy as np
import os

# Hàm để tìm số đếm lớn nhất trong tên tệp đã tồn tại
def get_new_filename(base_name, extension):
    count = 1
    while os.path.exists(f"{base_name}_{count}{extension}"):
        count += 1
    return f"{base_name}_{count}{extension}"

# Hàm nhúng ảnh watermark vào ảnh gốc bằng cách sử dụng LSB
def embed_watermark(image_path, watermark_path, output_path):
    # Mở ảnh gốc và ảnh watermark
    img = Image.open(image_path)
    watermark = Image.open(watermark_path)

    # Chuyển ảnh watermark sang grayscale nếu chưa phải
    watermark = watermark.convert('L')  # 'L' là mode ảnh grayscale

    # Kiểm tra nếu ảnh watermark nhỏ hơn hoặc bằng ảnh gốc
    if watermark.size[0] > img.size[0] or watermark.size[1] > img.size[1]:
        print("Ảnh watermark quá lớn so với ảnh gốc!")
        return

    # Chuyển ảnh gốc thành numpy array
    img_array = np.array(img)
    watermark_array = np.array(watermark)

    # Duyệt qua từng pixel của ảnh gốc và watermark
    for i in range(watermark.size[0]):  # Duyệt theo chiều rộng của watermark
        for j in range(watermark.size[1]):  # Duyệt theo chiều cao của watermark
            if i < img.size[0] and j < img.size[1]:  # Kiểm tra nếu pixel watermark nằm trong ảnh gốc
                # Lấy 1 bit từ watermark (đổi pixel grayscale thành bit)
                watermark_bit = (watermark_array[j, i] >> 7) & 1  # Lấy bit đầu tiên từ byte (pixel grayscale)

                # Nhúng bit vào mỗi kênh R, G, B của ảnh gốc
                r, g, b = img_array[j, i]
                r_new = (r & ~1) | watermark_bit  # Thay đổi bit ít quan trọng nhất của kênh R
                g_new = (g & ~1) | watermark_bit  # Thay đổi bit ít quan trọng nhất của kênh G
                b_new = (b & ~1) | watermark_bit  # Thay đổi bit ít quan trọng nhất của kênh B

                # Cập nhật pixel mới vào ảnh
                img_array[j, i] = [r_new, g_new, b_new]

    # Tạo tên tệp mới với số đếm nếu tệp đã tồn tại
    new_output_path = get_new_filename("watermarked_image", ".png")

    # Chuyển lại ảnh từ numpy array thành ảnh và lưu lại
    watermarked_img = Image.fromarray(img_array)
    watermarked_img.save(new_output_path)
    print(f"Ảnh đã được nhúng watermark và lưu tại: {new_output_path}")

# Hàm trích xuất watermark từ ảnh đã nhúng
def extract_watermark(image_path, watermark_size, output_path):
    # Mở ảnh đã nhúng watermark
    img = Image.open(image_path)
    img_array = np.array(img)

    # Tạo một mảng mới để chứa ảnh watermark
    watermark_array = np.zeros((watermark_size[1], watermark_size[0]), dtype=np.uint8)

    # Duyệt qua từng pixel của ảnh đã nhúng watermark và trích xuất LSB
    for i in range(watermark_size[0]):  # Duyệt theo chiều rộng của watermark
        for j in range(watermark_size[1]):  # Duyệt theo chiều cao của watermark
            if i < img.size[0] and j < img.size[1]:  # Kiểm tra nếu pixel watermark nằm trong ảnh gốc
                r, g, b = img_array[j, i]  # Lấy các kênh R, G, B của pixel
                watermark_bit = (r & 1)  # Lấy bit LSB của kênh R (có thể sử dụng kênh nào cũng được)
                watermark_array[j, i] = watermark_bit * 255  # Chuyển bit thành pixel grayscale (0 hoặc 255)

    # Tạo tên tệp mới với số đếm nếu tệp đã tồn tại
    new_extracted_path = get_new_filename("extracted_watermark", ".png")

    # Chuyển mảng watermark thành ảnh và lưu lại
    watermark_img = Image.fromarray(watermark_array)
    watermark_img.save(new_extracted_path)
    print(f"Ảnh watermark đã được trích xuất và lưu tại: {new_extracted_path}")

# Ví dụ sử dụng
image_path = 'IMG_9785.png'  # Đường dẫn đến ảnh gốc
watermark_path = 'miles.jpg'  # Đường dẫn đến ảnh watermark

# Nhúng watermark vào ảnh
embed_watermark(image_path, watermark_path, '')

# Trích xuất watermark từ ảnh đã nhúng (bạn cần cung cấp kích thước của watermark đã nhúng)
watermark_size = (225, 225)  # Kích thước của watermark (width, height) cần biết trước
extract_watermark('watermarked_image.png', watermark_size, '')

from PIL import Image
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim
import sys
import os

# Đảm bảo in ra tiếng Việt không bị lỗi
sys.stdout.reconfigure(encoding='utf-8')

def get_new_filename(base_name, extension):
    counter = 1
    new_filename = f"{base_name}{extension}"
    while os.path.exists(new_filename):
        new_filename = f"{base_name}_{counter}{extension}"
        counter += 1
    return new_filename


# --- Các hàm xử lý file văn bản ---
def read_text_file(txt_path):
    with open(txt_path, 'r', encoding='utf-8') as f:
        return f.read()

def write_text_file(txt_path, content):
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(content)

# --- Hàm chuyển đổi ---
def to_binary(data):
    return ''.join(format(ord(char), '08b') for char in data)

def from_binary(binary_data):
    chars = [binary_data[i:i+8] for i in range(0, len(binary_data), 8)]
    return ''.join([chr(int(char, 2)) for char in chars if int(char, 2) != 0])

# --- Kiểm tra khả năng nhúng ---
def can_embed_data(image_path, data):
    img = Image.open(image_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    num_pixels = img.width * img.height
    max_data_bits = num_pixels * 3
    data_bits = len(to_binary(data)) + 16  # marker
    print(f"Kích thước ảnh gốc: {img.width}x{img.height}, Dữ liệu cần nhúng: {len(to_binary(data))} bits.")
    return data_bits <= max_data_bits

# --- Nhúng dữ liệu văn bản ---
def embed_text_data(image_path, txt_path, output_path):
    if not os.path.exists(txt_path):
        print(f"[Lỗi] Không tìm thấy file văn bản: {txt_path}")
        return

    data = read_text_file(txt_path)
    if not can_embed_data(image_path, data):
        print("Dữ liệu quá dài để giấu vào ảnh này!")
        return

    img = Image.open(image_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')

    binary_data = to_binary(data) + '1111111111111110'  # marker EOF
    data_index = 0

    pixels = list(img.getdata())
    new_pixels = []

    for pixel in pixels:
        r, g, b = pixel
        if data_index < len(binary_data):
            r = (r & ~1) | int(binary_data[data_index])
            data_index += 1
        if data_index < len(binary_data):
            g = (g & ~1) | int(binary_data[data_index])
            data_index += 1
        if data_index < len(binary_data):
            b = (b & ~1) | int(binary_data[data_index])
            data_index += 1
        new_pixels.append((r, g, b))

    img.putdata(new_pixels)
    img.save(output_path)
    print(f"Đã nhúng dữ liệu văn bản vào ảnh và lưu tại: {output_path}")

# --- Nhúng dữ liệu hình ảnh ---
def embed_image_data(image_path, input_image_path, output_path):
    # Kiểm tra kích thước ảnh để nhúng vào
    img = Image.open(image_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Mở ảnh watermark cần nhúng
    input_img = Image.open(input_image_path)
    if input_img.mode != 'RGB':
        input_img = input_img.convert('RGB')

    # Mở ảnh cần nhúng vào
    input_img = Image.open(input_image_path)
    input_img = input_img.resize(img.size)  # Đảm bảo kích thước ảnh cần nhúng khớp với ảnh gốc

    input_pixels = list(input_img.getdata())
    img_pixels = list(img.getdata())

    new_pixels = []

    # Nhúng ảnh vào ảnh gốc (tương tự như chèn dữ liệu văn bản)
    data_index = 0
    for i in range(len(img_pixels)):
        r, g, b = img_pixels[i]
        input_r, input_g, input_b = input_pixels[i] 

        # Lấy bit của các kênh màu ảnh cần nhúng
        r = (r & ~1) | (input_r >> 7) 
        g = (g & ~1) | (input_g >> 7)
        b = (b & ~1) | (input_b >> 7)

        new_pixels.append((r, g, b))
    
    img.putdata(new_pixels)
    img.save(output_path)
    print(f"Đã nhúng hình ảnh vào ảnh và lưu tại: {output_path}")

# # --- Nhúng dữ liệu hình ảnh vào kênh R ---
# def embed_image_data(image_path, input_image_path, output_path):
#     # Mở ảnh gốc
#     img = Image.open(image_path)
#     if img.mode != 'RGB':
#         img = img.convert('RGB')
    
#     # Mở ảnh cần nhúng
#     input_img = Image.open(input_image_path).convert('L')  # Chuyển về grayscale
#     input_img = input_img.resize(img.size)

#     input_pixels = list(input_img.getdata())  # chỉ 1 giá trị grayscale
#     img_pixels = list(img.getdata())

#     new_pixels = []

#     for i in range(len(img_pixels)):
#         r, g, b = img_pixels[i]
#         watermark_bit = input_pixels[i] >> 7  # chỉ lấy bit cao nhất

#         r = (r & ~1) | watermark_bit  # nhúng vào LSB của R

#         new_pixels.append((r, g, b))

#     img.putdata(new_pixels)
#     img.save(output_path)
#     print(f"Đã nhúng watermark (1 bit) vào kênh R và lưu tại: {output_path}")

# --- Trích xuất dữ liệu ---
def extract_text(image_path, output_txt_path):
    img = Image.open(image_path)
    pixels = list(img.getdata())
    binary_data = ''

    for pixel in pixels:
        for channel in pixel:
            binary_data += str(channel & 1)

    eof_index = binary_data.find('1111111111111110')
    if eof_index != -1:
        binary_data = binary_data[:eof_index]

    result = from_binary(binary_data)
    write_text_file(output_txt_path, result)
    print(f"Dữ liệu trích xuất đã lưu vào: {output_txt_path}")
    return result

# Hàm trích xuất watermark từ ảnh đã nhúng (trích xuất từ 3 kênh R, G, B)
def extract_img(image_path, output_path):
    # Mở ảnh đã nhúng watermark
    img = Image.open(image_path)
    img_array = np.array(img)

    # Lấy kích thước của ảnh gốc
    img_width, img_height = img.size

    # Tạo một mảng mới để chứa ảnh watermark (màu sắc)
    watermark_array = np.zeros((img_height, img_width, 3), dtype=np.uint8)  # 3 kênh RGB

    # Duyệt qua từng pixel của ảnh đã nhúng watermark và trích xuất LSB từ cả 3 kênh
    for i in range(img_width):  # Duyệt chiều rộng ảnh
        for j in range(img_height):  # Duyệt chiều cao ảnh
            r, g, b = img_array[j, i]  # Lấy R, G, B của pixel

            # Lấy bit LSB của từng kênh R, G, B
            watermark_bit_r = (r & 1)
            watermark_bit_g = (g & 1)
            watermark_bit_b = (b & 1)

            # Gán giá trị cho từng kênh của ảnh watermark
            watermark_array[j, i] = (watermark_bit_r * 255, watermark_bit_g * 255, watermark_bit_b * 255)

    # Chuyển mảng watermark thành ảnh và lưu lại
    watermark_img = Image.fromarray(watermark_array)
    watermark_img.save(output_path)
    print(f"Ảnh watermark đã được trích xuất và lưu tại: {output_path}")

# # Hàm trích xuất watermark từ ảnh đã nhúng
# def extract_img(image_path, output_path):
#     # Mở ảnh đã nhúng watermark
#     img = Image.open(image_path)
#     img_array = np.array(img)

#     # Lấy kích thước của ảnh gốc
#     img_width, img_height = img.size

#     # Tạo một mảng mới để chứa ảnh watermark
#     watermark_array = np.zeros((img_height, img_width), dtype=np.uint8)

#     # Duyệt qua từng pixel của ảnh đã nhúng watermark và trích xuất LSB
#     for i in range(img_width):  # Duyệt theo chiều rộng của ảnh
#         for j in range(img_height):  # Duyệt theo chiều cao của ảnh
#             r, g, b = img_array[j, i]  # Lấy các kênh R, G, B của pixel
#             watermark_bit = (r & 1)  # Lấy bit LSB của kênh R (có thể sử dụng kênh nào cũng được)
#             watermark_array[j, i] = watermark_bit * 255  # Chuyển bit thành pixel grayscale (0 hoặc 255)

#     # Chuyển mảng watermark thành ảnh và lưu lại
#     watermark_img = Image.fromarray(watermark_array)
#     watermark_img.save(output_path)
#     print(f"Ảnh watermark đã được trích xuất và lưu tại: {output_path}")

# --- Tính PSNR & SSIM ---
def psnr(image1, image2):
    image1 = image1.astype(np.float64)
    image2 = image2.astype(np.float64)
    mse_value = np.mean((image1 - image2) ** 2)
    if mse_value == 0:
        return 100
    return 10 * np.log10((255.0 ** 2) / mse_value)

def ssim_index(image1, image2):
    return ssim(image1, image2, channel_axis=2)

def evaluate_quality(original_image_path, watermarked_image_path):
    original = cv2.imread(original_image_path)
    watermarked = cv2.imread(watermarked_image_path)

    if original.shape != watermarked.shape:
        print("Kích thước ảnh không khớp.")
        return

    psnr_value = psnr(original, watermarked)
    ssim_value = ssim_index(original, watermarked)

    print(f'PSNR: {psnr_value:.2f} dB')
    print(f'SSIM: {ssim_value:.4f}')

# ============ main ============

# Cấu hình đường dẫn
input_folder = 'input_images'
output_folder = 'output_images'
text_folder = 'text_files'
img_folder = 'watermark_img'

input_image_1 = 'dac-diem-kien-truc-ai-cap-co-dai.jpg'
# input_image_2 = 'IMG_9785.png'
input_image_2 = 'stock-photo-159533631-1500x1000.jpg'
output_image = 'watermarked.png'
input_text = 'message_to_embed.txt'
# input_watermark_image = 'miles.jpg'  # Ảnh cần nhúng vào
input_watermark_image = '1000_F_539348176_ulGRbIS9rDObiEfl4MFrbwKNXQCe6SZC.jpg'
output_text = 'extracted_message.txt'

# Tạo các đường dẫn đầy đủ
input_image_path = os.path.join(input_folder, input_image_2)
output_image_path = get_new_filename(os.path.join(output_folder, "watermarked"), ".png")
input_text_path = os.path.join(text_folder, input_text)
output_text_path = get_new_filename(os.path.join(text_folder, "extracted_message"), ".txt")
input_watermark_img_path = os.path.join(img_folder, input_watermark_image)
output_watermark_img_path = get_new_filename(os.path.join(img_folder, "extracted_watermark_img"), ".jpg")

# Hỏi người dùng lựa chọn
print("Chọn thao tác:")
print("1. Chèn văn bản vào ảnh")
print("2. Chèn hình ảnh vào ảnh")

choice = input("Nhập lựa chọn (1/2): ")

if choice == '1':
    # Chèn văn bản vào ảnh
    embed_text_data(input_image_path, input_text_path, output_image_path)
    # Sau khi nhúng xong, trích xuất dữ liệu đã nhúng
    extracted_text = extract_text(output_image_path, output_text_path)
    # Gọi hàm đánh giá chất lượng ảnh
    evaluate_quality(input_image_path, output_image_path)
    if extracted_text:
        print(f"Dữ liệu trích xuất: {extracted_text}")
elif choice == '2':
    # Chèn hình ảnh vào ảnh
    embed_image_data(input_image_path, input_watermark_img_path, output_image_path)
    # Sau khi nhúng xong, trích xuất dữ liệu đã nhúng (nếu có)
    extracted_img = extract_img(output_image_path, output_watermark_img_path)
    if extracted_img:
        print(f"Ảnh trích xuất thành công và lưu tại: {extracted_img}")
else:
    print("Lựa chọn không hợp lệ!")



import cv2
import numpy as np
from scipy.fftpack import dctn, idctn # Giữ nguyên thư viện DCT
import math
from skimage.metrics import structural_similarity as ssim # Giữ nguyên để đánh giá ảnh
import os
import traceback

# --- Các hằng số cấu hình ---
ALPHA = 10.0 # Cường độ nhúng (có thể cần điều chỉnh tùy ảnh và text)
BLOCK_SIZE = 8
COEFF_POSITION = (4, 4) # Vị trí hệ số DCT để sửa đổi (tần số trung bình)
LEN_BITS = 32 # Số bit dùng để lưu độ dài của văn bản (thường là 32 hoặc 64)

# --- Hàm chuẩn hóa DCT/IDCT ---
def dct2(block):
    """Applies 2D DCT Type II to a block with orthogonal normalization."""
    if block.dtype != np.float64: block = block.astype(np.float64)
    return dctn(block, type=2, norm='ortho', axes=[0, 1])

def idct2(block):
    """Applies 2D IDCT Type II to a block with orthogonal normalization."""
    if block.dtype != np.float64: block = block.astype(np.float64)
    return idctn(block, type=2, norm='ortho', axes=[0, 1])

# --- Hàm chuyển đổi văn bản/nhị phân ---
def text_to_binary(text):
    """Chuyển đổi chuỗi văn bản thành chuỗi nhị phân (UTF-8)."""
    try:
        encoded_bytes = text.encode('utf-8')
        binary_string = ''.join(format(byte, '08b') for byte in encoded_bytes)
        return binary_string
    except Exception as e:
        print(f"Lỗi khi chuyển văn bản sang nhị phân: {e}")
        traceback.print_exc()
        return None

def binary_to_text(binary_string):
    """Chuyển đổi chuỗi nhị phân thành văn bản (UTF-8)."""
    if not binary_string:
         print("Cảnh báo: Chuỗi nhị phân đầu vào rỗng.")
         return ""
    if len(binary_string) % 8 != 0:
        print(f"Cảnh báo: Chuỗi nhị phân không hợp lệ, độ dài {len(binary_string)} không chia hết cho 8.")
        valid_len = (len(binary_string) // 8) * 8
        print(f"    Sẽ chỉ sử dụng {valid_len} bit đầu tiên.")
        binary_string = binary_string[:valid_len]
        if not binary_string:
             return "[Lỗi giải mã - không còn bit hợp lệ sau khi cắt]"
    try:
        byte_list = [int(binary_string[i:i+8], 2) for i in range(0, len(binary_string), 8)]
        byte_data = bytes(byte_list)
        return byte_data.decode('utf-8', errors='replace')
    except ValueError as e:
        print(f"Lỗi khi chuyển đổi nhị phân sang byte (ValueError): {e}")
        traceback.print_exc()
        return "[Lỗi giải mã - giá trị nhị phân không hợp lệ]"
    except Exception as e:
        print(f"Lỗi không xác định khi giải mã văn bản: {e}")
        traceback.print_exc()
        return "[Lỗi giải mã không xác định]"

# --- Hàm nhúng thủy vân văn bản ---
def embed_text_watermark(host_image_path, text_to_embed, output_path,
                         block_size=BLOCK_SIZE, alpha=ALPHA, coeff_pos=COEFF_POSITION, len_bits=LEN_BITS):
    """
    Embeds text into a host image using DCT.
    Returns (image_data, original_binary_string) or (None, None)
    """
    original_binary_string_generated = None
    try:
        # ... (Đọc ảnh gốc) ...
        host_img = cv2.imread(host_image_path, cv2.IMREAD_GRAYSCALE)
        if host_img is None:
            print(f"Lỗi: Không thể đọc ảnh gốc từ {host_image_path}")
            return None, None
        print(f"Kích thước ảnh gốc: {host_img.shape}")

        # ... (Chuyển đổi văn bản thành bit & chuẩn bị chuỗi bit) ...
        binary_text = text_to_binary(text_to_embed)
        if binary_text is None:
             print("Lỗi: Không thể chuyển đổi văn bản thành nhị phân.")
             return None, None
        text_len_bits = len(binary_text)
        print(f"Độ dài văn bản gốc (bits): {text_len_bits}")
        len_binary_string = format(text_len_bits, f'0{len_bits}b')
        binary_watermark_with_len = len_binary_string + binary_text
        original_binary_string_generated = binary_watermark_with_len
        total_bits_to_embed = len(binary_watermark_with_len)
        print(f"Tổng số bit cần nhúng (bao gồm độ dài): {total_bits_to_embed}")
        print(f"Chuỗi nhị phân GỐC sẽ nhúng ({total_bits_to_embed} bits): {binary_watermark_with_len[:100]}...")

        # ... (Chuẩn bị ảnh gốc, padding, kiểm tra dung lượng) ...
        h, w = host_img.shape
        h_pad = (block_size - h % block_size) % block_size
        w_pad = (block_size - w % block_size) % block_size
        host_padded = cv2.copyMakeBorder(host_img, 0, h_pad, 0, w_pad, cv2.BORDER_CONSTANT, value=0)
        h_p, w_p = host_padded.shape
        num_blocks_h = h_p // block_size
        num_blocks_w = w_p // block_size
        total_blocks = num_blocks_h * num_blocks_w
        print(f"Kích thước ảnh sau padding: {host_padded.shape}")
        print(f"Tổng số khối {block_size}x{block_size} (khả năng chứa): {total_blocks} bits")
        if total_bits_to_embed > total_blocks:
            print(f"Lỗi: Văn bản quá dài ({total_bits_to_embed} bits) cho ảnh gốc (khả năng chứa: {total_blocks} bits).")
            return None, original_binary_string_generated # Trả về chuỗi bit gốc dù lỗi

        # ... (Vòng lặp nhúng chính) ...
        watermarked_img_float = host_padded.astype(np.float64).copy()
        watermark_bit_index = 0
        for i in range(num_blocks_h):
            for j in range(num_blocks_w):
                if watermark_bit_index >= total_bits_to_embed: break
                row_start, row_end = i * block_size, (i + 1) * block_size
                col_start, col_end = j * block_size, (j + 1) * block_size
                if row_end > h_p or col_end > w_p: continue
                block = watermarked_img_float[row_start:row_end, col_start:col_end]
                if block.shape != (block_size, block_size): continue

                dct_block = dct2(block)
                bit = int(binary_watermark_with_len[watermark_bit_index])
                if bit == 1:
                    dct_block[coeff_pos[0], coeff_pos[1]] += alpha
                idct_block = idct2(dct_block)
                watermarked_img_float[row_start:row_end, col_start:col_end] = idct_block
                watermark_bit_index += 1
            if watermark_bit_index >= total_bits_to_embed: break

        # ... (Hậu xử lý và Lưu ảnh) ...
        watermarked_img_clipped = np.clip(watermarked_img_float, 0, 255)
        watermarked_final = watermarked_img_clipped[0:h, 0:w]
        watermarked_final_uint8 = watermarked_final.astype(np.uint8)
        output_dir_check = os.path.dirname(output_path)
        if output_dir_check and not os.path.exists(output_dir_check):
                 os.makedirs(output_dir_check)
                 print(f"Đã tạo thư mục: {output_dir_check}")
        save_success = cv2.imwrite(output_path, watermarked_final_uint8)
        if save_success:
            print(f"Ảnh đã nhúng văn bản được lưu tại: {output_path}")
            return watermarked_final_uint8, original_binary_string_generated
        else:
            print(f"LỖI: Không thể lưu ảnh thủy vân vào {output_path}")
            return None, original_binary_string_generated # Trả về chuỗi bit gốc dù lỗi lưu

    except Exception as e:
        print(f"Đã xảy ra lỗi trong quá trình nhúng văn bản: {e}")
        traceback.print_exc()
        return None, original_binary_string_generated # Trả về chuỗi bit gốc nếu đã tạo

# --- Hàm trích xuất thủy vân văn bản ---
def extract_text_watermark(original_image_path, watermarked_image_path,
                           block_size=BLOCK_SIZE, alpha=ALPHA, coeff_pos=COEFF_POSITION, len_bits=LEN_BITS):
    """
    Extracts the embedded text using DCT (requires original image).
    Returns (extracted_text, relevant_extracted_bits) or (error_message, None)
    """
    extracted_text_result = "[Lỗi trích xuất - không xác định]"
    relevant_extracted_bits_result = None
    try:
        # ... (Đọc ảnh, xử lý kích thước, padding) ...
        original_img = cv2.imread(original_image_path, cv2.IMREAD_GRAYSCALE)
        watermarked_img = cv2.imread(watermarked_image_path, cv2.IMREAD_GRAYSCALE)
        if original_img is None or watermarked_img is None:
            print("Lỗi: Không thể đọc ảnh gốc hoặc ảnh đã thủy vân.")
            return "[Lỗi đọc ảnh]", None
        if original_img.shape != watermarked_img.shape:
            print(f"Cảnh báo: Kích thước ảnh gốc {original_img.shape} và ảnh thủy vân {watermarked_img.shape} không khớp. Thử cắt.")
            h_orig, w_orig = original_img.shape
            h_wm, w_wm = watermarked_img.shape
            if h_orig >= h_wm and w_orig >= w_wm:
                 original_img = original_img[:h_wm, :w_wm]
                 print(f"    Đã cắt ảnh gốc về kích thước: {original_img.shape}")
                 if original_img.shape != watermarked_img.shape:
                     print("Lỗi: Vẫn không khớp sau khi cắt.")
                     return "[Lỗi kích thước ảnh sau cắt]", None
            else:
                 print("Lỗi nghiêm trọng: Ảnh gốc nhỏ hơn ảnh thủy vân?")
                 return "[Lỗi kích thước ảnh không hợp lệ]", None
        h, w = original_img.shape
        h_pad = (block_size - h % block_size) % block_size
        w_pad = (block_size - w % block_size) % block_size
        original_padded = cv2.copyMakeBorder(original_img, 0, h_pad, 0, w_pad, cv2.BORDER_CONSTANT, value=0)
        watermarked_padded = cv2.copyMakeBorder(watermarked_img, 0, h_pad, 0, w_pad, cv2.BORDER_CONSTANT, value=0)
        if original_padded.shape != watermarked_padded.shape:
            print("Lỗi nghiêm trọng: Kích thước sau padding không khớp!")
            return "[Lỗi padding]", None
        h_p, w_p = original_padded.shape
        num_blocks_h = h_p // block_size
        num_blocks_w = w_p // block_size
        total_blocks = num_blocks_h * num_blocks_w

        # ... (Khởi tạo biến trích xuất) ...
        extracted_bits_with_len = ""
        message_len = -1
        bits_extracted_count = 0
        expected_total_bits = -1
        original_padded_float = original_padded.astype(np.float64)
        watermarked_padded_float = watermarked_padded.astype(np.float64)
        extraction_threshold = alpha / 2.0

        # ... (Vòng lặp trích xuất chính) ...
        stop_extraction = False
        for i in range(num_blocks_h):
            if stop_extraction: break
            for j in range(num_blocks_w):
                # Trích xuất 1 bit
                row_start, row_end = i * block_size, (i + 1) * block_size
                col_start, col_end = j * block_size, (j + 1) * block_size
                if row_end > h_p or col_end > w_p: continue
                block_orig = original_padded_float[row_start:row_end, col_start:col_end]
                block_wm = watermarked_padded_float[row_start:row_end, col_start:col_end]
                if block_orig.shape != (block_size, block_size) or block_wm.shape != (block_size, block_size): continue

                dct_orig = dct2(block_orig)
                dct_wm = dct2(block_wm)
                coeff_orig = dct_orig[coeff_pos[0], coeff_pos[1]]
                coeff_wm = dct_wm[coeff_pos[0], coeff_pos[1]]
                extracted_bit = "1" if coeff_wm > coeff_orig + extraction_threshold else "0"
                extracted_bits_with_len += extracted_bit
                bits_extracted_count += 1

                # Kiểm tra trạng thái
                if message_len == -1 and bits_extracted_count == len_bits:
                    try:
                        message_len = int(extracted_bits_with_len, 2)
                        print(f"Đã trích xuất độ dài tin nhắn: {message_len} bits")
                        expected_total_bits = len_bits + message_len
                        if message_len < 0:
                             extracted_text_result = "[Lỗi giải mã - độ dài âm]"
                             stop_extraction = True; break
                        if expected_total_bits > total_blocks:
                             extracted_text_result = "[Lỗi giải mã - độ dài quá lớn]"
                             stop_extraction = True; break
                        if message_len == 0:
                            extracted_text_result = ""
                            relevant_extracted_bits_result = extracted_bits_with_len[:len_bits]
                            stop_extraction = True; break
                    except ValueError:
                        extracted_text_result = "[Lỗi giải mã - không đọc được độ dài]"
                        stop_extraction = True; break
                elif message_len != -1 and bits_extracted_count >= expected_total_bits:
                     print(f"Đã trích xuất đủ {expected_total_bits} bits theo độ dài.")
                     stop_extraction = True; break
            # Kết thúc vòng lặp j
        # Kết thúc vòng lặp i

        # ... (Xử lý kết quả cuối cùng) ...
        if message_len == -1:
             if bits_extracted_count >= len_bits:
                 try:
                     message_len = int(extracted_bits_with_len[:len_bits], 2)
                     print(f"Đã trích xuất độ dài tin nhắn (cuối vòng lặp): {message_len} bits")
                     expected_total_bits = len_bits + message_len
                     if message_len < 0 or expected_total_bits > total_blocks:
                         extracted_text_result = "[Lỗi giải mã - độ dài không hợp lệ (cuối)]"
                         message_len = -1
                     elif message_len == 0:
                          extracted_text_result = ""
                          relevant_extracted_bits_result = extracted_bits_with_len[:len_bits]
                 except ValueError:
                     extracted_text_result = "[Lỗi giải mã - không đọc được độ dài (cuối)]"
                     message_len = -1
             else:
                 extracted_text_result = "[Lỗi giải mã - không đủ bit độ dài cuối]"
                 message_len = -1

        if message_len > 0:
            expected_total_bits = len_bits + message_len
            actual_data_bits_count = max(0, bits_extracted_count - len_bits)
            num_data_bits_to_decode = min(actual_data_bits_count, message_len)
            extracted_message_bits = extracted_bits_with_len[len_bits : len_bits + num_data_bits_to_decode]

            if bits_extracted_count < expected_total_bits:
                 print(f"Cảnh báo: Chỉ trích xuất được {bits_extracted_count} bits, cần {expected_total_bits}. Dữ liệu text có thể bị mất.")
            if not extracted_message_bits:
                 if extracted_text_result.startswith("[Lỗi"): pass # Giữ lỗi cũ nếu có
                 else: extracted_text_result = "[Lỗi trích xuất - không có bit dữ liệu]"
            else:
                 print(f"Chuỗi nhị phân của tin nhắn sẽ giải mã ({len(extracted_message_bits)} bits): {extracted_message_bits[:100]}...")
                 extracted_text_result = binary_to_text(extracted_message_bits)

            actual_extracted_len = min(len(extracted_bits_with_len), expected_total_bits if expected_total_bits > 0 else len_bits)
            relevant_extracted_bits_result = extracted_bits_with_len[:actual_extracted_len]
        elif message_len == 0 and relevant_extracted_bits_result is None: # Đảm bảo gán relevant_bits khi len=0
             relevant_extracted_bits_result = extracted_bits_with_len[:len_bits]

        print(f"Hoàn tất trích xuất. Tổng số bit đã đọc: {bits_extracted_count}")
        return extracted_text_result, relevant_extracted_bits_result

    except Exception as e:
        print(f"Đã xảy ra lỗi nghiêm trọng trong quá trình trích xuất văn bản: {e}")
        traceback.print_exc()
        return "[Lỗi hệ thống khi trích xuất]", None


# --- Các hàm đánh giá ---
def calculate_psnr(img1, img2):
    """Calculates the Peak Signal-to-Noise Ratio between two images."""
    if img1 is None or img2 is None: return -1.0
    if img1.shape != img2.shape:
         print(f"PSNR Error: Shape mismatch {img1.shape} vs {img2.shape}")
         return -1.0
    img1_f64 = img1.astype(np.float64)
    img2_f64 = img2.astype(np.float64)
    mse = np.mean((img1_f64 - img2_f64) ** 2)
    if mse < 1e-10: return float('inf')
    max_pixel = 255.0
    try:
        psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
    except (ValueError, ZeroDivisionError):
        psnr = -1.0
    return psnr

def calculate_ssim(img1, img2):
    """Calculates the Structural Similarity Index between two images."""
    if img1 is None or img2 is None: return -1.0
    if img1.shape != img2.shape:
         print(f"SSIM Error: Shape mismatch {img1.shape} vs {img2.shape}")
         return -1.0
    if len(img1.shape) > 2: img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    if len(img2.shape) > 2: img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    if img1.dtype != np.uint8: img1 = np.clip(img1, 0, 255).astype(np.uint8)
    if img2.dtype != np.uint8: img2 = np.clip(img2, 0, 255).astype(np.uint8)
    try:
        min_dim = min(img1.shape)
        win_size = min(7, min_dim if min_dim >= 3 and min_dim % 2 == 1 else (min_dim - 1 if min_dim >= 4 else 3))
        if win_size < 3:
             print("SSIM Warning: Image dimensions too small.")
             return 0.0
        ssim_value = ssim(img1, img2, data_range=255, win_size=win_size)
        return ssim_value
    except Exception as e:
        print(f"SSIM Error: {e}")
        traceback.print_exc()
        return -1.0

# --- ***** HÀM TÍNH BER MỚI CHO CHUỖI NHỊ PHÂN ***** ---
def calculate_ber_strings(original_bits, extracted_bits):
    """Calculates the Bit Error Rate between two binary strings."""
    if original_bits is None or extracted_bits is None:
        print("Lỗi BER: Một hoặc cả hai chuỗi bit là None.")
        return -1.0 # Lỗi
    if not original_bits: # Chuỗi gốc không được rỗng trừ khi đó là ý định
         print("Cảnh báo BER: Chuỗi bit gốc rỗng.")
         return 0.0 if not extracted_bits else 1.0 # BER=0 nếu cả hai rỗng, BER=1 nếu trích xuất có bit

    len_orig = len(original_bits)
    len_extr = len(extracted_bits)
    total_bits = len_orig # So sánh dựa trên độ dài gốc

    if total_bits == 0: # Trường hợp chuỗi gốc rỗng đã xử lý
         return 0.0

    error_bits = 0
    # Duyệt qua độ dài của chuỗi gốc
    for i in range(total_bits):
        # Nếu chuỗi trích xuất ngắn hơn -> bit lỗi (bit bị thiếu)
        if i >= len_extr:
            error_bits += 1
        # Nếu bit khác nhau -> bit lỗi
        elif original_bits[i] != extracted_bits[i]:
            error_bits += 1
        # Nếu giống nhau thì không làm gì

    ber = error_bits / total_bits
    return ber, error_bits, total_bits # Trả về thêm số lỗi và tổng số bit để in


# --- Ví dụ sử dụng ---
if __name__ == "__main__":
    # --- Thiết lập đường dẫn ---
    base_dir = r'D:\DigitalWatermarking\DCT-IDCT\data' # Sử dụng thư mục hiện tại
    input_dir = os.path.join(base_dir, 'input')
    output_dir = os.path.join(base_dir, 'output_text')

    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # --- Định nghĩa file ảnh gốc và file văn bản ---
    host_image_file = os.path.join(input_dir, 'image.png') # Đảm bảo có ảnh này
    text_file_path = os.path.join(input_dir, "text.txt")

    watermarked_image_file = os.path.join(output_dir, f'text_watermarked_dct_alpha{ALPHA}_pos{COEFF_POSITION[0]}_{COEFF_POSITION[1]}.png')

    # --- Tạo file mẫu nếu cần ---
    if not os.path.exists(host_image_file):
        print(f"Tạo ảnh gốc mẫu '{host_image_file}' (512x512 ảnh xám)")
        img_y, img_x = np.meshgrid(np.linspace(0, 255, 512), np.linspace(50, 200, 512))
        img_host_sample = ((img_x + img_y) / 2).astype(np.uint8)
        if not cv2.imwrite(host_image_file, img_host_sample):
             print(f"FATAL ERROR: Không thể ghi ảnh mẫu '{host_image_file}'")
             exit(1)

    text_content_to_embed = None
    try:
        if not os.path.exists(text_file_path):
            print(f"Tạo file văn bản mẫu '{text_file_path}'")
            try:
                with open(text_file_path, 'w', encoding='utf-8') as f:
                    f.write("Đây là nội dung tiếng Việt mẫu.\n")
                    f.write("This is sample English content.\n")
                    f.write("12345!@#$%\n")
                    f.write("テスト。\n")
                    f.write("😊👍")
            except IOError as e:
                 print(f"Lỗi IO khi tạo file văn bản mẫu: {e}")
                 text_file_path = None

        if text_file_path:
            with open(text_file_path, 'r', encoding='utf-8') as file:
                text_content_to_embed = file.read().strip()
            if not text_content_to_embed and text_content_to_embed != "": # Phân biệt file rỗng và lỗi đọc
                 print(f"Cảnh báo: File '{text_file_path}' có vẻ rỗng hoặc chỉ chứa khoảng trắng.")
                 text_content_to_embed = ""

            print(f"Đã đọc văn bản từ file: '{text_file_path}'")
            print(f"Nội dung cần nhúng: '{text_content_to_embed[:200]}...'")

    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file văn bản '{text_file_path}'")
        text_content_to_embed = None
    except Exception as e:
        print(f"Lỗi khi đọc file văn bản: {e}")
        traceback.print_exc()
        text_content_to_embed = None

    # --- Bắt đầu xử lý ---
    if text_content_to_embed is not None:
        print(f"\nSử dụng ALPHA = {ALPHA}, Vị trí hệ số = {COEFF_POSITION}, LEN_BITS = {LEN_BITS}")

        # --- Nhúng ---
        print("\n--- Bắt đầu nhúng thủy vân văn bản ---")
        watermarked_img_data, original_binary_string = embed_text_watermark(
            host_image_file, text_content_to_embed, watermarked_image_file,
            block_size=BLOCK_SIZE, alpha=ALPHA, coeff_pos=COEFF_POSITION, len_bits=LEN_BITS
        )

        # --- Đánh giá nhúng ---
        psnr_value = -1.0
        ssim_value = -1.0
        original_img_data = cv2.imread(host_image_file, cv2.IMREAD_GRAYSCALE)
        if watermarked_img_data is not None and original_img_data is not None:
            psnr_value = calculate_psnr(original_img_data, watermarked_img_data)
            print(f"PSNR giữa ảnh gốc và ảnh đã nhúng text: {psnr_value:.2f} dB")
            ssim_value = calculate_ssim(original_img_data, watermarked_img_data)
            print(f"SSIM giữa ảnh gốc và ảnh đã nhúng text: {ssim_value:.4f}")
        # ... (Xử lý lỗi đọc ảnh gốc hoặc lỗi nhúng) ...

        # --- Trích xuất ---
        print("\n--- Bắt đầu trích xuất thủy vân văn bản ---")
        extracted_text_data = None
        extracted_binary_string = None
        if os.path.exists(watermarked_image_file):
            extracted_text_data, extracted_binary_string = extract_text_watermark(
                host_image_file, watermarked_image_file,
                block_size=BLOCK_SIZE, alpha=ALPHA, coeff_pos=COEFF_POSITION, len_bits=LEN_BITS
            )

            # --- So sánh Kết quả ---
            if extracted_text_data is not None:
                print("\n--- So sánh kết quả ---")
                print(f"Văn bản gốc         : '{text_content_to_embed}'")
                print(f"Văn bản trích xuất  : '{extracted_text_data}'")

                # --- So sánh chuỗi nhị phân ---
                print("\n--- So sánh chuỗi nhị phân ---")
                if original_binary_string is not None and extracted_binary_string is not None:
                    # ... (In 100 bit đầu) ...
                    orig_len = len(original_binary_string)
                    extr_len = len(extracted_binary_string)
                    print(f"Nhị phân gốc      ({orig_len} bits): {original_binary_string[:100]}...")
                    print(f"Nhị phân trích xuất({extr_len} bits): {extracted_binary_string[:100]}...")

                    # === TÍNH VÀ IN BER ===
                    ber_value, error_bits, total_ber_bits = calculate_ber_strings(original_binary_string, extracted_binary_string)
                    if ber_value >= 0:
                         print(f"BER giữa chuỗi bit gốc và trích xuất: {ber_value:.6f} ({error_bits}/{total_ber_bits} errors)")
                    else:
                         print("Không thể tính BER (lỗi).")
                    # === KẾT THÚC TÍNH BER ===

                    # So sánh chi tiết chuỗi bit
                    if original_binary_string == extracted_binary_string:
                        print(">>> Chuỗi nhị phân khớp!")
                    elif orig_len > extr_len and original_binary_string.startswith(extracted_binary_string):
                         print(">>> CẢNH BÁO: Chuỗi nhị phân trích xuất là phần đầu của chuỗi gốc (trích xuất thiếu bit?).")
                    elif extr_len > orig_len and extracted_binary_string.startswith(original_binary_string):
                         print(">>> CẢNH BÁO: Chuỗi nhị phân gốc là phần đầu của chuỗi trích xuất (trích xuất thừa bit?).")
                    else:
                        print(">>> LỖI: Chuỗi nhị phân KHÔNG khớp.")
                        diff_bit_idx = -1
                        min_len = min(orig_len, extr_len)
                        for i in range(min_len):
                            if original_binary_string[i] != extracted_binary_string[i]:
                                diff_bit_idx = i
                                break
                        if diff_bit_idx == -1 and orig_len != extr_len: diff_bit_idx = min_len
                        print(f"    (Lỗi bit đầu tiên ở vị trí {diff_bit_idx})" if diff_bit_idx != -1 else "    (Lỗi không xác định)")
                # ... (Xử lý lỗi thiếu chuỗi bit) ...

                # --- So sánh văn bản (logic giữ nguyên) ---
                if not isinstance(extracted_text_data, str) or not extracted_text_data.startswith("[Lỗi"):
                    if extracted_text_data == text_content_to_embed:
                        print("\n>>> THÀNH CÔNG: Văn bản trích xuất khớp với văn bản gốc!")
                    else:
                        # ... (code tìm lỗi trong văn bản) ...
                         diff_idx = -1
                         min_len_text = min(len(text_content_to_embed), len(extracted_text_data))
                         for i in range(min_len_text):
                             if text_content_to_embed[i] != extracted_text_data[i]:
                                 diff_idx = i; break
                         if diff_idx == -1 and len(text_content_to_embed) != len(extracted_text_data): diff_idx = min_len_text
                         print(f"\n>>> THẤT BẠI: Văn bản trích xuất KHÔNG khớp. (Lỗi đầu tiên ở vị trí ~{diff_idx})" if diff_idx != -1 else "\n>>> THẤT BẠI: Văn bản trích xuất KHÔNG khớp (Độ dài khác nhau?).")
                else:
                     print("\n(Văn bản trích xuất có lỗi giải mã, không so sánh nội dung chi tiết)")

            else: # extracted_text_data is None
                print("Trích xuất thủy vân văn bản thất bại nghiêm trọng.")
        else: # file ảnh wm không tồn tại
            print(f"Ảnh thủy vân '{watermarked_image_file}' không tồn tại hoặc nhúng ban đầu thất bại. Bỏ qua trích xuất.")

        # --- Hiển thị ảnh ---
        if original_img_data is not None and watermarked_img_data is not None:
            try:
                # ... (code hiển thị ảnh giữ nguyên) ...
                scale_percent = 50
                width_orig = int(original_img_data.shape[1] * scale_percent / 100)
                height_orig = int(original_img_data.shape[0] * scale_percent / 100)
                width_orig = max(1, width_orig); height_orig = max(1, height_orig)
                dim_orig = (width_orig, height_orig)
                resized_orig = cv2.resize(original_img_data, dim_orig, interpolation = cv2.INTER_AREA)
                resized_wm = cv2.resize(watermarked_img_data, dim_orig, interpolation = cv2.INTER_AREA)
                cv2.imshow('1. Original Host', resized_orig)
                cv2.imshow('2. Watermarked Image (Text Embedded)', resized_wm)
                print("\nNhấn phím bất kỳ trên cửa sổ ảnh để đóng.")
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            except Exception as display_error:
                print(f"\nKhông thể hiển thị ảnh: {display_error}")
        else:
            print("\nẢnh gốc hoặc ảnh thủy vân không tồn tại, không thể hiển thị.")

    else: # text_content_to_embed is None
        print("\nKhông đọc được nội dung văn bản từ file, không thể thực hiện nhúng/trích xuất.")

    print("\n--- Kết thúc chương trình ---")
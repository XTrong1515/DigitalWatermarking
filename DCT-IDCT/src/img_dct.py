import cv2
import numpy as np
from scipy.fftpack import dctn, idctn
import math
from skimage.metrics import structural_similarity as ssim
import os # Thêm để kiểm tra thư mục
import traceback # Thêm để in lỗi chi tiết hơn

ALPHA = 0.5  # === THAY ĐỔI: Cường độ nhúng  ===
# Cảnh báo: Alpha quá nhỏ có thể làm BER tăng vọt!
BLOCK_SIZE = 8
# COEFF_POSITION = (1, 1) # Vị trí cũ
COEFF_POSITION = (4, 4) # === THAY ĐỔI: Thử vị trí hệ số khác (tần số trung bình) ===
# Bạn có thể thử các cặp (2,3), (3,2), (4,4) etc. Tránh (0,0) (DC coeff) và các hệ số tần số quá cao.

# --- Hàm chuẩn hóa DCT/IDCT (Type 2) ---
def dct2(block):
    """Applies 2D DCT Type II to a block with orthogonal normalization."""
    return dctn(block, type=2, norm='ortho', axes=[0, 1])

def idct2(block):
    """Applies 2D IDCT Type II to a block with orthogonal normalization."""
    return idctn(block, type=2, norm='ortho', axes=[0, 1])

# --- Hàm nhúng thủy vân ---
def embed_watermark(host_image_path, watermark_image_path, output_path,
                    block_size=BLOCK_SIZE, alpha=ALPHA, coeff_pos=COEFF_POSITION):
    """
    Embeds a watermark into a host image using DCT.
    (Đã chỉnh sửa vị trí clip)
    """
    try:
        # 1. Đọc ảnh
        host_img = cv2.imread(host_image_path, cv2.IMREAD_GRAYSCALE)
        watermark_img = cv2.imread(watermark_image_path, cv2.IMREAD_GRAYSCALE)

        if host_img is None:
            print(f"Lỗi: Không thể đọc ảnh gốc từ {host_image_path}")
            return None
        if watermark_img is None:
            print(f"Lỗi: Không thể đọc ảnh thủy vân từ {watermark_image_path}")
            return None

        print(f"Kích thước ảnh gốc: {host_img.shape}")
        print(f"Kích thước thủy vân: {watermark_img.shape}")

        # 2. Tiền xử lý thủy vân
        _, watermark_binary = cv2.threshold(watermark_img, 127, 255, cv2.THRESH_BINARY)
        watermark_bits = (watermark_binary.flatten() / 255).astype(int)
        num_watermark_bits = len(watermark_bits)
        print(f"Số lượng bit thủy vân: {num_watermark_bits}")

        # 3. Chuẩn bị ảnh gốc
        h, w = host_img.shape
        h_pad = (block_size - h % block_size) % block_size
        w_pad = (block_size - w % block_size) % block_size
        host_padded = cv2.copyMakeBorder(host_img, 0, h_pad, 0, w_pad, cv2.BORDER_CONSTANT, value=0)
        h_p, w_p = host_padded.shape
        num_blocks_h = h_p // block_size
        num_blocks_w = w_p // block_size
        total_blocks = num_blocks_h * num_blocks_w
        print(f"Kích thước ảnh sau padding: {host_padded.shape}")
        print(f"Tổng số khối {block_size}x{block_size}: {total_blocks}")

        if num_watermark_bits > total_blocks:
            print(f"Lỗi: Thủy vân quá lớn ({num_watermark_bits} bits) cho ảnh gốc ({total_blocks} khối).")
            watermark_bits = watermark_bits[:total_blocks]
            num_watermark_bits = total_blocks
            print(f"Cảnh báo: Thủy vân đã bị cắt còn {num_watermark_bits} bits.")

        # Làm việc trên ảnh float để tránh mất mát độ chính xác
        watermarked_img_float = host_padded.astype(np.float64).copy() # Sử dụng float64 để tăng độ chính xác
        watermark_bit_index = 0

        # 4. Lặp qua các khối và nhúng
        for i in range(num_blocks_h):
            for j in range(num_blocks_w):
                if watermark_bit_index >= num_watermark_bits:
                    break

                row_start, row_end = i * block_size, (i + 1) * block_size
                col_start, col_end = j * block_size, (j + 1) * block_size
                block = watermarked_img_float[row_start:row_end, col_start:col_end]

                dct_block = dct2(block)
                bit = watermark_bits[watermark_bit_index]

                # Sửa đổi hệ số DCT chọn lọc
                # Cân nhắc: có thể thử nghiệm sửa đổi cả khi bit = 0 (ví dụ: -= alpha)
                # Hoặc các phương pháp nhúng khác phức tạp hơn.
                if bit == 1:
                    dct_block[coeff_pos[0], coeff_pos[1]] += alpha
                # Tùy chọn: elif bit == 0: dct_block[coeff_pos] -= alpha # Nhúng đối xứng

                idct_block = idct2(dct_block)

                # === THAY ĐỔI: KHÔNG clip ở đây ===
                # idct_block = np.clip(idct_block, 0, 255) # <--- BỎ DÒNG NÀY

                watermarked_img_float[row_start:row_end, col_start:col_end] = idct_block
                watermark_bit_index += 1
            if watermark_bit_index >= num_watermark_bits:
                break

        # 5. Hậu xử lý và Lưu ảnh
        # === THAY ĐỔI: Clip toàn bộ ảnh MỘT LẦN ở cuối ===
        watermarked_img_clipped = np.clip(watermarked_img_float, 0, 255)

        # Cắt bỏ padding
        watermarked_final = watermarked_img_clipped[0:h, 0:w]
        # Chuyển đổi kiểu dữ liệu sang uint8 để lưu
        watermarked_final_uint8 = watermarked_final.astype(np.uint8)

        # Kiểm tra thư mục output tồn tại chưa
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
             os.makedirs(output_dir)
             print(f"Đã tạo thư mục: {output_dir}")

        save_success = cv2.imwrite(output_path, watermarked_final_uint8)
        if save_success:
            print(f"Ảnh đã thủy vân được lưu tại: {output_path}")
            return watermarked_final_uint8 # Trả về dữ liệu ảnh uint8
        else:
            print(f"LỖI: Không thể lưu ảnh thủy vân vào {output_path}")
            return None

    except Exception as e:
        print(f"Đã xảy ra lỗi trong quá trình nhúng: {e}")
        traceback.print_exc()
        return None

# --- Hàm trích xuất thủy vân (Yêu cầu ảnh gốc) ---
def extract_watermark(original_image_path, watermarked_image_path,
                      watermark_shape, # Kích thước gốc của thủy vân (h, w)
                      block_size=BLOCK_SIZE, alpha=ALPHA, coeff_pos=COEFF_POSITION):
    """
    Extracts the watermark using DCT (requires original image).
    (Đã sửa đổi để dùng float64)
    """
    try:
        # 1. Đọc ảnh
        original_img = cv2.imread(original_image_path, cv2.IMREAD_GRAYSCALE)
        watermarked_img = cv2.imread(watermarked_image_path, cv2.IMREAD_GRAYSCALE)

        if original_img is None:
            print(f"Lỗi: Không thể đọc ảnh gốc từ {original_image_path}")
            return None
        if watermarked_img is None:
            print(f"Lỗi: Không thể đọc ảnh đã thủy vân từ {watermarked_image_path}")
            return None

        if original_img.shape != watermarked_img.shape:
            print(f"Lỗi: Kích thước ảnh gốc {original_img.shape} và ảnh thủy vân {watermarked_img.shape} không khớp.")
            # Cân nhắc: Resize ảnh thủy vân nếu cần? Hoặc báo lỗi nghiêm trọng hơn.
            # return None # Tạm thời cho phép chạy tiếp nếu chỉ padding khác nhau
            h_orig, w_orig = original_img.shape
            h_wm, w_wm = watermarked_img.shape
            if h_orig > h_wm or w_orig > w_wm:
                 print("Lỗi nghiêm trọng: Ảnh gốc lớn hơn ảnh thủy vân?")
                 return None
            # Giả sử ảnh gốc là chưa padding, ảnh wm đã padding và cắt lại
            original_img = original_img[:h_wm, :w_wm] # Cắt ảnh gốc cho khớp (RỦI RO)
            print(f"Cảnh báo: Kích thước không khớp, đã thử cắt ảnh gốc về {original_img.shape}")
            if original_img.shape != watermarked_img.shape:
                 print("Lỗi: Vẫn không khớp sau khi cắt.")
                 return None


        # 2. Chuẩn bị ảnh (padding phải giống lúc nhúng)
        h, w = original_img.shape
        h_pad = (block_size - h % block_size) % block_size
        w_pad = (block_size - w % block_size) % block_size
        original_padded = cv2.copyMakeBorder(original_img, 0, h_pad, 0, w_pad, cv2.BORDER_CONSTANT, value=0)
        watermarked_padded = cv2.copyMakeBorder(watermarked_img, 0, h_pad, 0, w_pad, cv2.BORDER_CONSTANT, value=0)

        # Kiểm tra padding có tạo ra kích thước khớp không
        if original_padded.shape != watermarked_padded.shape:
             print(f"Lỗi nghiêm trọng: Kích thước sau padding không khớp! Gốc: {original_padded.shape}, WM: {watermarked_padded.shape}")
             return None

        h_p, w_p = original_padded.shape
        num_blocks_h = h_p // block_size
        num_blocks_w = w_p // block_size

        extracted_bits = []
        num_bits_to_extract = watermark_shape[0] * watermark_shape[1]

        # Sử dụng float64 để tăng độ chính xác khi tính toán DCT
        original_padded_float = original_padded.astype(np.float64)
        watermarked_padded_float = watermarked_padded.astype(np.float64)


        # 3. Lặp qua các khối và trích xuất
        bit_count = 0
        # Ngưỡng trích xuất - Rất quan trọng, đặc biệt với alpha nhỏ
        extraction_threshold = alpha / 2.0
        # Cân nhắc: Có thể cần ngưỡng thích nghi hơn dựa trên độ lớn hệ số gốc?

        for i in range(num_blocks_h):
            for j in range(num_blocks_w):
                if bit_count >= num_bits_to_extract:
                    break

                row_start, row_end = i * block_size, (i + 1) * block_size
                col_start, col_end = j * block_size, (j + 1) * block_size

                block_orig = original_padded_float[row_start:row_end, col_start:col_end]
                block_wm = watermarked_padded_float[row_start:row_end, col_start:col_end]

                dct_orig = dct2(block_orig)
                dct_wm = dct2(block_wm)

                coeff_orig = dct_orig[coeff_pos[0], coeff_pos[1]]
                coeff_wm = dct_wm[coeff_pos[0], coeff_pos[1]]

                # Logic trích xuất: So sánh sự thay đổi với ngưỡng
                # Với alpha rất nhỏ, sự khác biệt coeff_wm - coeff_orig có thể cực nhỏ
                # và dễ bị nhiễu.
                if coeff_wm > coeff_orig + extraction_threshold:
                    extracted_bits.append(1)
                # elif coeff_wm < coeff_orig - extraction_threshold: # Nếu dùng nhúng đối xứng
                #    extracted_bits.append(0)
                else: # Bao gồm cả trường hợp gần bằng hoặc nhỏ hơn (ứng với bit 0)
                    extracted_bits.append(0)

                bit_count += 1
            if bit_count >= num_bits_to_extract:
                break

        # 4. Tái tạo ảnh thủy vân
        if len(extracted_bits) < num_bits_to_extract:
            print(f"Cảnh báo: Chỉ trích xuất được {len(extracted_bits)} bits, cần {num_bits_to_extract}. Padding phần còn lại bằng 0.")
            extracted_bits.extend([0] * (num_bits_to_extract - len(extracted_bits)))

        extracted_watermark_flat = np.array(extracted_bits[:num_bits_to_extract])
        # Reshape và chuyển sang uint8 với giá trị 0 hoặc 255
        extracted_watermark_img = (extracted_watermark_flat.reshape(watermark_shape) * 255).astype(np.uint8)

        print(f"Đã trích xuất {len(extracted_bits)} bits.")
        return extracted_watermark_img

    except Exception as e:
        print(f"Đã xảy ra lỗi trong quá trình trích xuất: {e}")
        traceback.print_exc()
        return None

# --- Hàm tính PSNR ---
def calculate_psnr(img1, img2):
    """Calculates the Peak Signal-to-Noise Ratio between two images."""
    if img1 is None or img2 is None:
        print("Lỗi PSNR: Một hoặc cả hai ảnh là None.")
        return -1

    # Chuyển đổi an toàn sang kiểu float64 trước khi tính toán
    if img1.shape != img2.shape:
         print(f"Lỗi PSNR: Kích thước ảnh không khớp. {img1.shape} vs {img2.shape}")
         # Thử resize cái nhỏ hơn? Hoặc cắt cái lớn hơn? (Rủi ro)
         # Tạm thời trả về lỗi
         return -1

    # Đảm bảo cả hai là float64 để tính MSE chính xác
    img1_f64 = img1.astype(np.float64)
    img2_f64 = img2.astype(np.float64)

    mse = np.mean((img1_f64 - img2_f64) ** 2)
    if mse == 0:
        return float('inf') # Hoàn toàn giống nhau
    max_pixel = 255.0
    try:
        psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
    except ValueError:
        psnr = -1 # Trường hợp mse âm hoặc không hợp lệ (không nên xảy ra)
    return psnr


# --- Hàm tính SSIM (Structural Similarity Index) ---
def calculate_ssim(img1, img2):
    if img1 is None or img2 is None:
        print("Lỗi SSIM: Một hoặc cả hai ảnh là None.")
        return -1
    if img1.shape != img2.shape:
        print(f"Lỗi SSIM: Kích thước ảnh không khớp. {img1.shape} vs {img2.shape}")
        return -1

    # Đảm bảo là uint8 cho skimage.metrics.ssim
    if img1.dtype != np.uint8:
        img1 = np.clip(img1, 0, 255).astype(np.uint8)
    if img2.dtype != np.uint8:
        img2 = np.clip(img2, 0, 255).astype(np.uint8)

    try:
        # data_range là giá trị lớn nhất có thể của pixel
        ssim_value, _ = ssim(img1, img2, data_range=img1.max() - img1.min(), full=True) # Sử dụng data_range động hơn
        # Hoặc đơn giản là: ssim_value = ssim(img1, img2, data_range=255) nếu bạn chắc chắn về khoảng giá trị
        return ssim_value
    except Exception as e:
        print(f"Lỗi khi tính SSIM: {e}")
        traceback.print_exc()
        return -1

# --- Hàm tính BER (Bit Error Rate) ---
def calculate_ber(original_watermark, extracted_watermark):
    """Calculates the Bit Error Rate between two binary watermarks."""
    if original_watermark is None or extracted_watermark is None:
         print("Lỗi BER: Một hoặc cả hai thủy vân là None.")
         return -1
    if original_watermark.shape != extracted_watermark.shape:
        print(f"Lỗi BER: Kích thước thủy vân không khớp. Gốc: {original_watermark.shape}, Trích xuất: {extracted_watermark.shape}")
        return -1

    # Đảm bảo cả hai là nhị phân 0/1
    # Ngưỡng 127 để chuyển ảnh xám (0-255) thành nhị phân (0 hoặc 1)
    wm1_bits = (original_watermark.flatten() > 127).astype(int)
    wm2_bits = (extracted_watermark.flatten() > 127).astype(int)

    if len(wm1_bits) == 0:
         print("Lỗi BER: Thủy vân không có bit nào.")
         return -1 # Hoặc 0.0 tùy ngữ cảnh

    error_bits = np.sum(wm1_bits != wm2_bits)
    total_bits = len(wm1_bits)

    ber = error_bits / total_bits
    return ber

# --- Ví dụ sử dụng ---
if __name__ == "__main__":
    # --- Thiết lập đường dẫn ---
    base_dir = r'D:\DigitalWatermarking\DCT-IDCT\data' # Thay đổi nếu cần
    input_dir = os.path.join(base_dir, 'input')
    output_dir = os.path.join(base_dir, 'output')

    # Đảm bảo thư mục tồn tại
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # host_image_file = os.path.join(input_dir, 'Lena-Original-Image-512x512-pixels.png')
    host_image_file = os.path.join(input_dir, 'image2.png') # 512x512 ảnh xám
    watermark_image_file = os.path.join(input_dir, 'watermark.png') # Thủy vân 64x64
    watermarked_image_file = os.path.join(output_dir, f'watermarked_dct_alpha{ALPHA}_pos{COEFF_POSITION[0]}_{COEFF_POSITION[1]}.png')
    extracted_watermark_file = os.path.join(output_dir, f'extracted_watermark_dct_alpha{ALPHA}_pos{COEFF_POSITION[0]}_{COEFF_POSITION[1]}.png')

    # --- Tạo ảnh mẫu nếu chưa có ---
    if not os.path.exists(host_image_file):
        print(f"Tạo ảnh gốc mẫu '{host_image_file}' (512x512 ảnh xám)")
        # Sử dụng ảnh gradient đơn giản thay vì random để dễ nhìn hơn
        img_y, img_x = np.meshgrid(np.linspace(0, 255, 512), np.linspace(50, 200, 512))
        img_host_sample = ((img_x + img_y) / 2).astype(np.uint8)
        cv2.imwrite(host_image_file, img_host_sample)

    if not os.path.exists(watermark_image_file):
        print(f"Tạo ảnh thủy vân mẫu '{watermark_image_file}' (64x64 nhị phân)")
        img_wm_sample = np.zeros((64, 64), dtype=np.uint8)
        # Vẽ chữ 'WM' lớn hơn
        cv2.putText(img_wm_sample, 'WM', (2, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255), 3)
        cv2.imwrite(watermark_image_file, img_wm_sample)
    # -----------------------------

    print(f"\nSử dụng ALPHA = {ALPHA}, Vị trí hệ số = {COEFF_POSITION}")

    # --- Nhúng thủy vân ---
    print("\n--- Bắt đầu nhúng thủy vân ---")
    watermarked_img_data = embed_watermark(host_image_file, watermark_image_file, watermarked_image_file,
                                           block_size=BLOCK_SIZE, alpha=ALPHA, coeff_pos=COEFF_POSITION)

    # --- Đánh giá chất lượng ảnh sau nhúng ---
    psnr_value = -1
    ssim_value = -1
    original_img_data = cv2.imread(host_image_file, cv2.IMREAD_GRAYSCALE)

    if watermarked_img_data is not None and original_img_data is not None:
        psnr_value = calculate_psnr(original_img_data, watermarked_img_data)
        print(f"PSNR giữa ảnh gốc và ảnh thủy vân: {psnr_value:.2f} dB")
        ssim_value = calculate_ssim(original_img_data, watermarked_img_data)
        print(f"SSIM giữa ảnh gốc và ảnh thủy vân: {ssim_value:.4f}") # SSIM gần 1 là rất tốt
    elif original_img_data is None:
         print("Không thể đọc lại ảnh gốc để tính PSNR/SSIM.")
    elif watermarked_img_data is None:
         print("Nhúng thủy vân thất bại, không thể tính PSNR/SSIM.")


    # --- Trích xuất thủy vân ---
    print("\n--- Bắt đầu trích xuất thủy vân ---")
    extracted_watermark_data = None
    original_watermark_data = cv2.imread(watermark_image_file, cv2.IMREAD_GRAYSCALE)

    if original_watermark_data is not None:
        watermark_original_shape = original_watermark_data.shape
        # Chỉ trích xuất nếu ảnh thủy vân đã được tạo thành công
        if os.path.exists(watermarked_image_file):
            extracted_watermark_data = extract_watermark(host_image_file, watermarked_image_file,
                                                         watermark_original_shape,
                                                         block_size=BLOCK_SIZE, alpha=ALPHA, coeff_pos=COEFF_POSITION)

            if extracted_watermark_data is not None:
                save_success = cv2.imwrite(extracted_watermark_file, extracted_watermark_data)
                if save_success:
                     print(f"Thủy vân trích xuất được lưu tại: {extracted_watermark_file}")
                else:
                     print(f"LỖI: Không thể lưu thủy vân trích xuất tại {extracted_watermark_file}")


                # --- Đánh giá chất lượng trích xuất ---
                psnr_extracted_value = calculate_psnr(original_watermark_data, extracted_watermark_data)
                print(f"PSNR giữa thủy vân gốc và trích xuất: {psnr_extracted_value:.2f} dB")

                ber_value = calculate_ber(original_watermark_data, extracted_watermark_data)
                print(f"Bit Error Rate (BER) giữa thủy vân gốc và trích xuất: {ber_value:.4f}")

            else:
                print("Trích xuất thủy vân thất bại.")
        else:
             print(f"Ảnh thủy vân '{watermarked_image_file}' không tồn tại. Bỏ qua trích xuất.")
    else:
        print(f"Không thể đọc ảnh thủy vân gốc '{watermark_image_file}' để lấy kích thước.")

    # --- Hiển thị ảnh (tùy chọn) ---
    if original_img_data is not None and original_watermark_data is not None \
       and watermarked_img_data is not None and extracted_watermark_data is not None:
        try:
            # Thay đổi kích thước để dễ xem hơn nếu cần
            scale_percent = 50 # percent of original size
            width_orig = int(original_img_data.shape[1] * scale_percent / 100)
            height_orig = int(original_img_data.shape[0] * scale_percent / 100)
            dim_orig = (width_orig, height_orig)

            width_wm = int(original_watermark_data.shape[1] * 200 / 100) # Phóng to thủy vân
            height_wm = int(original_watermark_data.shape[0] * 200 / 100)
            dim_wm = (width_wm, height_wm)


            cv2.imshow('1. Original Host', cv2.resize(original_img_data, dim_orig, interpolation = cv2.INTER_AREA))
            cv2.imshow('2. Watermark (Original)', cv2.resize(original_watermark_data, dim_wm, interpolation = cv2.INTER_NEAREST)) # Dùng INTER_NEAREST cho ảnh nhị phân
            cv2.imshow('3. Watermarked Image', cv2.resize(watermarked_img_data, dim_orig, interpolation = cv2.INTER_AREA))
            cv2.imshow('4. Extracted Watermark', cv2.resize(extracted_watermark_data, dim_wm, interpolation = cv2.INTER_NEAREST)) # Dùng INTER_NEAREST

            print("\nNhấn phím bất kỳ trên cửa sổ ảnh để đóng.")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception as display_error:
            print(f"\nKhông thể hiển thị ảnh (có thể do chạy trên môi trường không có GUI): {display_error}")
            # traceback.print_exc()
    else:
         print("\nMột số ảnh không tồn tại, không thể hiển thị.")
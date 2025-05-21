# -*- coding: utf-8 -*-
import cv2
import numpy as np
import os
import argparse # Thư viện để nhận tham số từ dòng lệnh
import math


def analyze_pixel_difference(original_image_path, watermarked_image_path,
                             output_dir, report_file_path=None, # Thêm tham số đường dẫn file report
                             num_sample_pixels=20, diff_amp_factor=10):
    """
    Phân tích sự khác biệt pixel, in tóm tắt ra console,
    ghi chi tiết ra file report (nếu được cung cấp), và lưu ảnh khác biệt.
    """
    console_lines = [] # Lưu các dòng in ra console
    report_lines = [] # Lưu các dòng ghi vào file report

    header1 = f"--- Bắt đầu phân tích sự khác biệt pixel ---"
    header2 = f"Ảnh gốc        : {original_image_path}"
    header3 = f"Ảnh đã nhúng   : {watermarked_image_path}"
    console_lines.extend([header1, header2, header3])
    report_lines.extend([header1, header2, header3])

    # --- 1. Đọc ảnh ---
    original_img = cv2.imread(original_image_path, cv2.IMREAD_GRAYSCALE)
    watermarked_img = cv2.imread(watermarked_image_path, cv2.IMREAD_GRAYSCALE)

    # --- 2. Kiểm tra lỗi cơ bản ---
    if original_img is None:
        error_msg = f"Lỗi: Không thể đọc ảnh gốc từ '{original_image_path}'"
        print(error_msg)
        return
    if watermarked_img is None:
        error_msg = f"Lỗi: Không thể đọc ảnh đã nhúng từ '{watermarked_image_path}'"
        print(error_msg)
        return
    if original_img.shape != watermarked_img.shape:
        error_msg = f"Lỗi: Kích thước ảnh gốc {original_img.shape} và ảnh nhúng {watermarked_img.shape} không khớp."
        print(error_msg)
        return

    shape_info = f"Kích thước ảnh    : {original_img.shape}"
    console_lines.append(shape_info)
    report_lines.append(shape_info)

    # --- 3. Tính toán thống kê khác biệt ---
    stats_header = "\n--- Thống kê sự thay đổi ---"
    console_lines.append(stats_header)
    report_lines.append(stats_header)

    diff_abs = cv2.absdiff(original_img, watermarked_img)
    changed_pixels_mask = (diff_abs > 0)
    num_changed = np.sum(changed_pixels_mask)
    total_pixels = original_img.size
    percent_changed = (num_changed / total_pixels) * 100 if total_pixels > 0 else 0
    max_diff = np.max(diff_abs) if num_changed > 0 else 0
    avg_diff_changed = np.mean(diff_abs[changed_pixels_mask]) if num_changed > 0 else 0
    avg_diff_total = np.mean(diff_abs)

    stats_lines = [
        f"Tổng số pixel: {total_pixels}",
        f"Số pixel bị thay đổi: {num_changed}",
        f"Tỷ lệ pixel thay đổi: {percent_changed:.4f}%",
        f"Mức chênh lệch tuyệt đối lớn nhất: {max_diff}",
        f"Mức chênh lệch tuyệt đối trung bình (chỉ trên các pixel thay đổi): {avg_diff_changed:.4f}",
        f"Mức chênh lệch tuyệt đối trung bình (trên toàn bộ ảnh): {avg_diff_total:.4f}"
    ]
    console_lines.extend(stats_lines) # Thống kê luôn in ra console
    report_lines.extend(stats_lines)  # Ghi cả vào report

    # --- 4. Tạo và lưu ảnh khác biệt ---
    difference_image_file = os.path.join(output_dir, f'difference_{os.path.basename(watermarked_image_path)}')
    diff_visual = cv2.convertScaleAbs(diff_abs, alpha=diff_amp_factor)
    diff_save_msg = ""
    try:
        os.makedirs(output_dir, exist_ok=True)
        save_diff_success = cv2.imwrite(difference_image_file, diff_visual)
        if save_diff_success:
            diff_save_msg = f"\nẢnh khác biệt (đã khuếch đại x{diff_amp_factor}) được lưu tại: {difference_image_file}"
        else:
            diff_save_msg = f"\nLỖI: Không thể lưu ảnh khác biệt tại {difference_image_file}"
    except Exception as e:
        diff_save_msg = f"\nLỗi khi lưu hoặc tạo thư mục cho ảnh khác biệt: {e}"

    console_lines.append(diff_save_msg)
    report_lines.append(diff_save_msg)

    # --- 5. Tạo dữ liệu bảng mẫu ---
    sample_table_header = f"\n--- Bảng so sánh mẫu ({num_sample_pixels} pixel đầu tiên) ---"
    separator = "-" * 60
    table_header = f"{'Tọa độ':<10} | {'Gốc':<5} | {'Nhúng':<5} | {'Chênh Lệch':<12} | {'Thay Đổi?':<10}"

    # Thêm bảng mẫu vào report_lines (luôn luôn)
    report_lines.append(sample_table_header)
    report_lines.append(separator)
    report_lines.append(table_header)
    report_lines.append(separator)

    count = 0
    rows, cols = original_img.shape
    for r in range(rows):
        for c in range(cols):
            if count >= num_sample_pixels:
                break
            orig_val = original_img[r, c]
            wm_val = watermarked_img[r, c]
            diff_signed = int(wm_val) - int(orig_val)
            changed_mark = "✔" if diff_signed != 0 else ""
            table_line = f"({r},{c}){'':<4} | {orig_val:<5} | {wm_val:<5} | {diff_signed:<12} | {changed_mark:<10}"
            report_lines.append(table_line) # Ghi dòng vào report
            count += 1
        if count >= num_sample_pixels:
            break
    report_lines.append(separator)

    # --- 6. In ra console ---
    for line in console_lines:
        print(line)
    # Chỉ in bảng mẫu ra console nếu không ghi ra file report
    if report_file_path is None:
        print(sample_table_header)
        print(separator)
        print(table_header)
        print(separator)
        # In lại các dòng bảng đã tạo
        table_data_start_index = report_lines.index(separator) + 3 # Tìm vị trí bắt đầu dữ liệu bảng
        for i in range(table_data_start_index, table_data_start_index + num_sample_pixels):
             if i < len(report_lines): print(report_lines[i])
        print(separator)


    # --- 7. Ghi ra file report (nếu có đường dẫn) ---
    if report_file_path:
        try:
            # Đảm bảo thư mục chứa file report tồn tại
            report_dir = os.path.dirname(report_file_path)
            if report_dir and not os.path.exists(report_dir):
                os.makedirs(report_dir)
                print(f"\nĐã tạo thư mục cho file report: {report_dir}")

            with open(report_file_path, 'w', encoding='utf-8') as f:
                for line in report_lines:
                    f.write(line + '\n') # Ghi từng dòng và thêm newline
            print(f"\n>>> Báo cáo phân tích chi tiết đã được ghi vào file: {report_file_path}")
        except Exception as e:
            print(f"\nLỖI: Không thể ghi báo cáo vào file '{report_file_path}': {e}")

# --- Xử lý tham số dòng lệnh và gọi hàm phân tích ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phân tích sự khác biệt pixel giữa ảnh gốc và ảnh đã nhúng thủy vân.")
    parser.add_argument("--original", required=True, help="Đường dẫn đến file ảnh gốc.")
    parser.add_argument("--watermarked", required=True, help="Đường dẫn đến file ảnh đã nhúng thủy vân.")
    parser.add_argument("--output-dir", default=".", help="Thư mục để lưu ảnh khác biệt (mặc định là thư mục hiện tại).")
    # === THÊM THAM SỐ FILE REPORT ===
    parser.add_argument("--report-file", help="(Tùy chọn) Đường dẫn đến file .txt để ghi báo cáo phân tích chi tiết.")
    parser.add_argument("--sample-size", type=int, default=20, help="Số lượng pixel mẫu trong báo cáo (mặc định: 20).")
    parser.add_argument("--amp", type=int, default=10, help="Hệ số khuếch đại ảnh khác biệt (mặc định: 10).")

    args = parser.parse_args()

    if not os.path.exists(args.original):
        print(f"Lỗi: File ảnh gốc không tồn tại: {args.original}")
    elif not os.path.exists(args.watermarked):
        print(f"Lỗi: File ảnh đã nhúng không tồn tại: {args.watermarked}")
    else:
        analyze_pixel_difference(args.original,
                                 args.watermarked,
                                 args.output_dir,
                                 args.report_file, # Truyền đường dẫn file report vào hàm
                                 args.sample_size,
                                 args.amp)
        print("\n--- Phân tích hoàn tất ---")

        # Đóng cửa sổ ảnh nếu có (để kết thúc chương trình sạch sẽ)
        try:
             cv2.waitKey(1000) # Chờ 1 giây để người dùng kịp nhìn
             cv2.destroyAllWindows()
        except:
             pass
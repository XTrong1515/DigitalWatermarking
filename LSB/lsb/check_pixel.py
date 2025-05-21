from PIL import Image

def get_lsb(pixel):
    """Trích xuất bit LSB từ một pixel RGB"""
    return tuple(value & 1 for value in pixel)

def compare_pixels(original_path, embedded_path, num_pixels=10, save_to_file=True):
    """
    So sánh các pixel giữa ảnh gốc và ảnh đã nhúng bằng phương pháp LSB.
    - original_path: đường dẫn ảnh gốc
    - embedded_path: đường dẫn ảnh sau khi nhúng
    - num_pixels: số pixel đầu tiên cần so sánh
    - save_to_file: có lưu kết quả ra file không
    """
    img_original = Image.open(original_path).convert('RGB')
    img_embedded = Image.open(embedded_path).convert('RGB')

    original_pixels = list(img_original.getdata())
    embedded_pixels = list(img_embedded.getdata())

    num_pixels = min(num_pixels, len(original_pixels))
    output_lines = []

    header = f"{'Pixel #':<8}{'Gốc (RGB)':<18}{'Nhúng (RGB)':<18}{'Δ (R,G,B)':<18}{'LSB Gốc':<10}{'LSB Nhúng':<12}{'Thay đổi'}"
    output_lines.append(header)
    output_lines.append('-' * len(header))

    for i in range(num_pixels):
        o_pixel = original_pixels[i]
        e_pixel = embedded_pixels[i]
        delta = tuple(e - o for o, e in zip(o_pixel, e_pixel))

        lsb_orig = get_lsb(o_pixel)
        lsb_embed = get_lsb(e_pixel)

        changed = '✔️' if o_pixel != e_pixel else ''
        line = f"{i:<8}{o_pixel!s:<18}{e_pixel!s:<18}{delta!s:<18}{str(lsb_orig):<10}{str(lsb_embed):<12}{changed}"
        output_lines.append(line)

    # In ra console
    for line in output_lines:
        print(line)

    # Ghi ra file nếu cần
    if save_to_file:
        with open("pixel_diff_report.txt", "w", encoding="utf-8") as f:
            for line in output_lines:
                f.write(line + "\n")
        print("\n Kết quả đã được lưu vào: pixel_diff_report.txt")

compare_pixels(
    r"C:\HocTap\BMMMT\digital_watermark\lsb\watermark_img\1000_F_539348176_ulGRbIS9rDObiEfl4MFrbwKNXQCe6SZC.jpg",
    r"C:\HocTap\BMMMT\digital_watermark\lsb\watermark_img\extracted_watermark_img_8.jpg",
    num_pixels=20
)



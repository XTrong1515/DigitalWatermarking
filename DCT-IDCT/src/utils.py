import cv2
import matplotlib.pyplot as plt

def load_image(image_path, grayscale=True):
    if grayscale:
        return cv2.imread(image_path, cv2.IMREAD_GRAYSCALE).astype(float)
    return cv2.imread(image_path)

def save_image(image, output_path):
    cv2.imwrite(output_path, image)

def display_results(original, watermark, watermarked, extracted):
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 4, 1)
    plt.title("Ảnh gốc")
    plt.imshow(original, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 4, 2)
    plt.title("Thủy vân gốc")
    plt.imshow(watermark, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 4, 3)
    plt.title("Ảnh có thủy vân")
    plt.imshow(watermarked, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 4, 4)
    plt.title("Thủy vân trích xuất")
    plt.imshow(extracted, cmap='gray')
    plt.axis('off')

    plt.show()
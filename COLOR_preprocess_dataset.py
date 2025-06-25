import os
import cv2
import shutil
from pathlib import Path
from PIL import Image
import numpy as np

def preprocess_image(image_path, output_dir, method=None):
    with Image.open(image_path) as img:
        img.verify()
    img = cv2.imread(image_path)

    if method == 'binary':
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, processed_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif method == 'adaptive_binary':
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        processed_img = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    else:
        raise ValueError("Метод должен быть 'grayscale', 'canny', 'binary' или 'blur_binary'")

    filename = os.path.basename(image_path)
    output_path = os.path.join(output_dir, filename)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    success = cv2.imwrite(output_path, processed_img)
    print(f"Сохранено: {output_path}")
    return success


def process_dataset(input_dir, output_dir, method=None):
    """
    Обрабатывает весь датасет, создавая новую структуру папок (train и val).
    """
    output_dir = Path(output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    for subdir in ['train', 'val']:
        input_subdir = os.path.join(input_dir, subdir)
        output_subdir = output_dir / subdir
        output_subdir.mkdir()

        for class_name in os.listdir(input_subdir):
            class_path = os.path.join(input_subdir, class_name)
            if os.path.isdir(class_path):
                output_class_path = output_subdir / class_name
                output_class_path.mkdir()
                print(f"Обработка класса {class_name} в {subdir}")
                for img_name in os.listdir(class_path):
                    img_path = os.path.join(class_path, img_name)
                    if img_path.lower().endswith(('.png')):
                        preprocess_image(img_path, str(output_class_path), method)

if __name__ == "__main__":
    input_dir = r"C:\Users\maria\OneDrive\Documents\job\extract_data\dataset"
    output_dir = r"C:\Users\maria\OneDrive\Documents\job\extract_data\dataset_processed_adaptive_binary"
    method = 'adaptive_binary'
    process_dataset(input_dir, output_dir, method)
    print(f"Датасет обработан и сохранён в {output_dir}")
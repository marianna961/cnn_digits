"""
Скрипт для организации и подготовки датасета
"""
import os
import shutil
import json
import random

def prepare_dataset(labeled_json="labeled_digits.json", input_folder="new_cropped_imgs", output_folder="dataset"):
    with open(labeled_json, 'r') as f:
        labeled_data = json.load(f)

    # Создаём папки для тренировочного и валидационного наборов
    train_folder = os.path.join(output_folder, "train")
    val_folder = os.path.join(output_folder, "val")
    classes = [str(i) for i in range(10)] + ["unknown"]
    
    for folder in [train_folder, val_folder]:
        for cls in classes:
            os.makedirs(os.path.join(folder, cls), exist_ok=True)

    # Список всех видео
    video_list = list(labeled_data.keys())
    random.seed(42)  # Фиксируем seed для воспроизводимости
    random.shuffle(video_list)

    # Разделение видео: 80% в train, 20% в val
    split_idx = int(len(video_list) * 0.8)
    train_videos = video_list[:split_idx]
    val_videos = video_list[split_idx:]

    # Функция для определения целевой папки на основе видео
    def get_dest_folder(video_name):
        return train_folder if video_name in train_videos else val_folder

    # Распределяем изображения по классам
    copied_files = 0
    skipped_files = 0

    for video_name in labeled_data:
        dest_folder_base = get_dest_folder(video_name)
        print(f"\nОбработка видео: {video_name} -> {dest_folder_base}")

        for region_type in ["date", "time"]:
            for frame_idx in labeled_data[video_name][region_type]:
                for pos_idx, label in labeled_data[video_name][region_type][frame_idx].items():
                    base_filename = f"{region_type}_char_frame_{frame_idx}_pos_{pos_idx}.png"
                    unique_filename = f"{video_name}_{base_filename}"
                    img_path = os.path.join(input_folder, video_name, base_filename)

                    if not os.path.exists(img_path):
                        print(f"Пропуск: Файл не найден: {img_path}")
                        skipped_files += 1
                        continue

                    # Определяем целевую папку на основе класса
                    label = "unknown" if label == "unknown" else str(label)
                    dest_path = os.path.join(dest_folder_base, label, unique_filename)

                    try:
                        shutil.copy(img_path, dest_path)
                        print(f"Скопировано: {img_path} -> {dest_path}")
                        copied_files += 1
                    except Exception as e:
                        print(f"Ошибка при копировании {img_path} -> {dest_path}: {e}")
                        skipped_files += 1

    print(f"\nДатасет подготовлен: скопировано {copied_files} файлов, пропущено {skipped_files} файлов")
    print(f"Папка датасета: {output_folder}")
    print(f"Видео в train: {len(train_videos)}, видео в val: {len(val_videos)}")

if __name__ == "__main__":
    prepare_dataset(
        labeled_json=r"C:\Users\maria\OneDrive\Documents\job\extract_data\labeled_digits.json",
        input_folder=r"C:\Users\maria\OneDrive\Documents\job\extract_data\new_cropped_imgs",
        output_folder=r"C:\Users\maria\OneDrive\Documents\job\extract_data\dataset"
    )
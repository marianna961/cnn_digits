"""
Скрипт для автоматической разметки цифр для подсчёта метрик.
Автоматически заполняет дату (13022025) и время, начиная с 13:30:08,
с увеличением на frame_interval секунд для каждого кадра.
Входные данные:
- Путь к видео и camera_params.json.
- Количество кадров для разметки (например, каждые 10 секунд).
Логика:
- Извлечение кадров с заданным интервалом, начиная с 10-й секунды.
- Автоматическое заполнение даты (13022025) и времени (HH:MM:SS).
- Сохранение результата в labeled_digits.json.
Выход:
- JSON-файл с разметкой, совместимый с print_metrics.py.
"""

import os
import json
import cv2

def load_camera_params(params_path, video_name):
    """
    Загружает параметры камеры для указанного видео из JSON.
    """
    with open(params_path, 'r') as f:
        all_params = json.load(f)
    base_name = os.path.splitext(video_name)[0]
    params = all_params.get(base_name)
    if not params:
        raise ValueError(f"Параметры для {base_name} не найдены в {params_path}")
    return params

def extract_frame(video_path, sec):
    """
    Извлекает кадр из видео на заданной секунде.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Не удалось открыть видео {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        raise ValueError(f"Не удалось определить FPS для видео {video_path}")
    frame_num = int(sec * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise ValueError(f"Не удалось извлечь кадр на {sec} секунде")
    # print(f"FPS видео: {fps}, кадр {frame_num} для {sec} секунды")
    return frame

def get_time_labels(start_hours, start_minutes, start_seconds, offset_seconds):
    """
    Вычисляет метки времени (HH:MM:SS) для заданного смещения в секундах.
    Возвращает словарь с метками для позиций 0–5.
    """
    total_seconds = start_hours * 3600 + start_minutes * 60 + start_seconds + offset_seconds
    hours = (total_seconds // 3600) % 24
    minutes = (total_seconds // 60) % 60
    seconds = total_seconds % 60
    
    time_str = f"{hours:02d}{minutes:02d}{seconds:02d}"  # Формат: HHMMSS
    return {
        "0": time_str[0],  # Часы: первая цифра
        "1": time_str[1],  # Часы: вторая цифра
        "2": time_str[2],  # Минуты: первая цифра
        "3": time_str[3],  # Минуты: вторая цифра
        "4": time_str[4],  # Секунды: первая цифра
        "5": time_str[5]   # Секунды: вторая цифра
    }

def label_digits(video_path, params_path, output_json, num_frames, frame_interval=10, start_sec=10):
    """
    Автоматически размечает дату (13022025) и время (начиная с 13:30:08) для кадров,
    сохраняет в JSON. Начинает с start_sec (по умолчанию 10 секунд).
    """
    video_name = os.path.basename(video_path)
    params = load_camera_params(params_path, video_name)

    labeled_data = {}
    labeled_data[video_name] = {"date": {}, "time": {}}

    # Начальное время: 13:30:08
    start_hours, start_minutes, start_seconds = 13, 36, 13

    for i in range(num_frames):
        sec = start_sec + i * frame_interval
        frame = extract_frame(video_path, sec)
        # print(f"\nРазметка кадра для {video_name} на {sec} секунде:")

        # Автоматическое заполнение даты: 13022025
        date_labels = {
            "0": "0",
            "1": "2",
            "2": "1",
            "3": "3",
            "4": "2",
            "5": "0",
            "6": "2",
            "7": "5"
        }
        labeled_data[video_name]["date"][str(sec)] = date_labels
        # print(f"Разметка даты для {sec} секунды (автоматическая): {date_labels}")

        # Автоматическое заполнение времени
        offset_seconds = i * frame_interval  # Смещение от начального времени
        time_labels = get_time_labels(start_hours, start_minutes, start_seconds, offset_seconds)
        labeled_data[video_name]["time"][str(sec)] = time_labels
        print(f"Разметка времени для {sec} секунды (автоматическая): {time_labels}")

    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(labeled_data, f, ensure_ascii=False, indent=4)
    print(f"Разметка сохранена в {output_json}")

if __name__ == "__main__":
    video_path = r"C:\Users\maria\OneDrive\Documents\job\extract_data\sip_12022025\sip_12022025\cam_3245_53_1739435410_0.mp4"
    params_path = r"C:\Users\maria\OneDrive\Documents\job\extract_data\configs\camera_params.json"
    output_json = r"C:\Users\maria\OneDrive\Documents\job\extract_data\metrics_params\dop_cam_3245_53_1739435410_0.json"
    num_frames = 530 # 50 кадров для стабильных метрик
    frame_interval = 1
    start_sec = 370

    label_digits(video_path, params_path, output_json, num_frames, frame_interval, start_sec)
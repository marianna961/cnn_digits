import cv2
import json
import numpy as np
import onnxruntime as ort
from PIL import Image
import os
from torchvision import transforms
import time

def load_camera_params(params_path, video_name):
    """
    Загружает параметры камеры для указанного видео из JSON.
    """
    with open(params_path, 'r') as f:
        all_params = json.load(f)
    base_name = os.path.splitext(video_name)[0]
    params = all_params.get(base_name)
    if not params:
        print(f"Параметры для {base_name} не найдены в {params_path}")
    return params

def preprocess_image(image, img_size=(32, 32), method=None):
    """
    Предобрабатывает изображение для модели.
    """
    if image.size == 0:
        print("Пустое изображение для предобработки")
        return None
    if method == 'binary':
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, processed_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif method == 'adaptive_binary':
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        processed_img = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    else:
        print(f"Неподдерживаемый метод: {method}")
        return None

    image = Image.fromarray(processed_img)
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    return transform(image).unsqueeze(0).numpy()

def predict_digits(roi_images, session, class_map, expected_digits, method=None):
    """
    Предсказывает цифры для списка изображений ROI.
    """
    predictions = []
    for img in roi_images:
        input_data = preprocess_image(img, method=method)
        if input_data is None:
            predictions.append('?')
            continue
        outputs = session.run(None, {'input': input_data})[0]
        pred_class = np.argmax(outputs, axis=1)[0]
        predictions.append(class_map.get(pred_class, '?'))

    if len(predictions) == expected_digits:
        return ''.join(predictions)
    else:
        return "0"

def validate_roi(x, y, w, h, max_width, max_height):
    """
    Проверяет, что ROI находится в допустимых границах.
    """
    x = max(0, min(x, max_width))
    y = max(0, min(y, max_height))
    w = max(1, min(w, max_width - x))
    h = max(1, min(h, max_height - y))
    return x, y, w, h

def format_time(digits):
    """
    Форматирует строку цифр как время (HH:MM:SS).
    """
    if len(digits) == 6:
        return f"{digits[0:2]}:{digits[2:4]}:{digits[4:6]}"
    return "0"

def format_date(digits):
    if len(digits) == 8:
        return f"{digits[0:2]}/{digits[2:4]}/{digits[4:8]}"
    return "0"

def process_video(input_video, output_video, onnx_model_path, params_path, img_size=(32, 32), method=None):

    video_name = os.path.basename(input_video)
    params = load_camera_params(params_path, video_name)

    cap = cv2.VideoCapture(input_video)

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Видео: {video_name}, {width}x{height}, {total_frames} кадров")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    session = ort.InferenceSession(onnx_model_path)
    class_map = {i: str(i) for i in range(10)}

    time_crop = params['time_crop_region']
    time_digits = params['time_digit_coords']['digits']
    video_res = params['video_resolution']
    video_width, video_height = video_res['width'], video_res['height']

    time_x_start = int(time_crop['x_start'] * video_width)
    time_y_start = int(time_crop['y_start'] * video_height)
    time_width = int(time_crop['width'] * video_width)
    time_height = int(time_crop['height'] * video_height)
    time_x_start, time_y_start, time_width, time_height = validate_roi(
        time_x_start, time_y_start, time_width, time_height, video_width, video_height)
    print(f"time ROI: time_x_start={time_x_start}, time_y_start={time_y_start}, time_width={time_width}, time_height={time_height}")

    date_crop = params.get('data_crop_region', {})
    date_digits = params.get('data_digit_coords', {}).get('digits', [])
    date_x_start = int(date_crop.get('x_start', 0) * video_width) if date_crop else 0
    date_y_start = int(date_crop.get('y_start', 0) * video_height) if date_crop else 0
    date_width = int(date_crop.get('width', 0) * video_width) if date_crop else 0
    date_height = int(date_crop.get('height', 0) * video_height) if date_crop else 0
    date_x_start, date_y_start, date_width, date_height = validate_roi(
        date_x_start, date_y_start, date_width, date_height, video_width, video_height)
    print(f"date roi: date_x_start={date_x_start}, date_y_start={date_y_start}, date_width={date_width}, date_height={date_height}")

    # spead
    frame_count = 0
    total_time = 0.0
    frame_times = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        start_time = time.time()

        # roi времени
        time_roi = frame[time_y_start:time_y_start + time_height, time_x_start:time_x_start + time_width]
        time_str = "0"
        if time_roi.size == 0:
            print(f"Пустой time_roi в кадре {frame_count}")
        else:
            time_digit_images = []
            for i, digit in enumerate(time_digits):
                x1 = int(digit['x1'] * time_width)
                y1 = int(digit['y1'] * time_height)
                w = int(digit['width'] * time_width)
                h = int(digit['height'] * time_height)
                x1, y1, w, h = validate_roi(x1, y1, w, h, time_width, time_height)
                digit_img = time_roi[y1:y1 + h, x1:x1 + w]
                if digit_img.size > 0:
                    time_digit_images.append(digit_img)
            digit_string = predict_digits(time_digit_images, session, class_map, expected_digits=6, method=method)
            time_str = format_time(digit_string)

        # Извлечение ROI даты
        date_str = "0"
        if date_crop and date_digits:
            date_roi = frame[date_y_start:date_y_start + date_height, date_x_start:date_x_start + date_width]
            if date_roi.size == 0:
                print("error empty roi")
            else:
                date_digit_images = []
                for i, digit in enumerate(date_digits):
                    x1 = int(digit['x1'] * date_width)
                    y1 = int(digit['y1'] * date_height)
                    w = int(digit['width'] * date_width)
                    h = int(digit['height'] * date_height)
                    x1, y1, w, h = validate_roi(x1, y1, w, h, date_width, date_height)
                    digit_img = date_roi[y1:y1 + h, x1:x1 + w]
                    if digit_img.size > 0:
                        date_digit_images.append(digit_img)
                digit_string = predict_digits(date_digit_images, session, class_map, expected_digits=8, method=method)
                date_str = format_date(digit_string)

        cv2.putText(frame, time_str, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, date_str, (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
        out.write(frame)

        frame_time = time.time() - start_time
        frame_times.append(frame_time)
        total_time += frame_time

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # time metrics
    avg_frame_time = total_time / frame_count if frame_count > 0 else 0
    avg_fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
    min_frame_time = min(frame_times) if frame_times else 0
    max_frame_time = max(frame_times) if frame_times else 0
    # min_fps = 1.0 / max_frame_time if max_frame_time > 0 else 0
    # max_fps = 1.0 / min_frame_time if min_frame_time > 0 else 0

    output_dir = os.path.dirname(output_video)
    speed_metrics_file = os.path.join(output_dir, f"{os.path.splitext(video_name)[0]}_speed_metrics.txt")
    with open(speed_metrics_file, 'w', encoding='utf-8') as f:
        f.write(f"Видео: {video_name}\n")
        f.write(f"Общее количество кадров: {frame_count}\n")
        f.write(f"Общее время обработки: {total_time:.2f} секунд\n")
        f.write(f"Среднее время на кадр: {avg_frame_time:.4f} секунд\n")
        f.write(f"Средний FPS: {avg_fps:.2f} кадров/сек\n")
        f.write(f"Минимальное время на кадр: {min_frame_time:.4f} секунд\n")
        f.write(f"Максимальное время на кадр: {max_frame_time:.4f} секунд\n")

    return {
        'video_name': video_name,
        'total_frames': frame_count,
        'total_time': total_time,
        'avg_fps': avg_fps,
    }

def process_videos(input_folder, output_folder, onnx_model_path, params_path, img_size=(32, 32), method=None):
    """
    Обрабатывает все видео из входной папки и сохраняет результаты в выходную папку.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    results = []
    for filename in os.listdir(input_folder):
        if filename.lower().endswith('.mp4'):
            input_video = os.path.join(input_folder, filename)
            output_video = os.path.join(output_folder, os.path.splitext(filename)[0] + '_annotated.mp4')
            print(f"Обработка видео: {input_video} -> {output_video}")
            result = process_video(input_video, output_video, onnx_model_path, params_path, img_size, method)
            results.append(result)

    # Сохранение сводных результатов
    summary_file = os.path.join(output_folder, "speed_summary.txt")
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("Сводные результаты измерения скорости:\n")
        for result in results:
            f.write(f"\nВидео: {result['video_name']}\n")
            f.write(f"Общее количество кадров: {result['total_frames']}\n")
            f.write(f"Общее время обработки: {result['total_time']:.2f} секунд\n")
            f.write(f"Средний FPS: {result['avg_fps']:.2f} кадров/сек\n")
            f.write(f"Минимальный FPS: {result['min_fps']:.2f} кадров/сек\n")
            f.write(f"Максимальный FPS: {result['max_fps']:.2f} кадров/сек\n")
    print(f"Сводные результаты сохранены в {summary_file}")

if __name__ == "__main__":
    input_folder = r"C:\Users\maria\OneDrive\Documents\job\extract_data\sip_12022025\sip_12022025"
    output_folder = r"C:\Users\maria\OneDrive\Documents\job\extract_data\demo\TIME_40e_resnet18_adaptive_binary"
    onnx_model_path = r"C:\Users\maria\OneDrive\Documents\job\extract_data\project\40e_resnet18_adaptive_binary.onnx"
    params_path = r"C:\Users\maria\OneDrive\Documents\job\extract_data\configs\camera_params.json"
    method = 'adaptive_binary'

    process_videos(input_folder, output_folder, onnx_model_path, params_path, method=method)
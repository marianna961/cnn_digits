import os
import json
import cv2
import numpy as np
import onnxruntime as ort
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from pathlib import Path
from PIL import Image
from torchvision import transforms

def load_json(file_path):
    """Загружает JSON-файл."""
    with open(file_path, 'r') as f:
        return json.load(f)

def preprocess_image(image, img_size=(32, 32), method=None):
    """Предобрабатывает изображение для модели с использованием указанного метода."""
    if image.size == 0:
        print("Пустое изображение для предобработки")
        return None
    if method == 'canny':
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        processed_img = cv2.Canny(gray_img, 100, 200)
    elif method == 'binary':
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, processed_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif method == 'blur_binary':
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray_img, (5, 5), 0)
        _, processed_img = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif method == 'adaptive_binary':
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        processed_img = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    elif method == 'Nothing':
        processed_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Как в process_video.py
    else:
        raise ValueError("Метод должен быть 'canny', 'binary', 'blur_binary' или 'adaptive_binary'")

    image = Image.fromarray(processed_img)
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5))  # Нормализация [-1, 1]
    ])
    image = transform(image).unsqueeze(0).numpy()
    return image

def predict_digit(session, img):
    """Предсказывает цифру с помощью ONNX модели."""
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    pred = session.run([output_name], {input_name: img})[0]
    return np.argmax(pred, axis=1)[0]

def validate_roi(x, y, w, h, max_width, max_height):
    """Проверяет, что ROI находится в допустимых границах."""
    x = max(0, min(x, max_width))
    y = max(0, min(y, max_height))
    w = max(1, min(w, max_width - x))
    h = max(1, min(h, max_height - y))
    return x, y, w, h

def evaluate_metrics(true_labels, pred_labels, position):
    """Вычисляет метрики для указанной позиции цифры."""
    accuracy = accuracy_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels, average='weighted', zero_division=0)
    precision = precision_score(true_labels, pred_labels, average='weighted', zero_division=0)
    recall = recall_score(true_labels, pred_labels, average='weighted', zero_division=0)
    report = classification_report(true_labels, pred_labels, digits=4, zero_division=0)
    print(f"\nМетрики для позиции {position}:")
    print(f"Точность (accuracy): {accuracy:.4f}")
    print(f"F1-мера (weighted): {f1:.4f}")
    print(f"Precision (weighted): {precision:.4f}")
    print(f"Recall (weighted): {recall:.4f}")
    print("Подробный отчёт:")
    print(report)
    return {'accuracy': accuracy, 'f1': f1, 'precision': precision, 'recall': recall, 'report': report}

def process_video(video_path, camera_params, metrics_data, onnx_session, output_dir, method=None):
    """Обрабатывает видео, предсказывает цифры и вычисляет метрики."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Не удалось открыть видео: {video_path}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        print(f"Ошибка: не удалось определить FPS для видео {video_path}")
        cap.release()
        return None
    print(f"FPS видео {video_path}: {fps}")

    video_name = os.path.splitext(os.path.basename(video_path))[0] + '.mp4'
    camera_key = video_name.replace('.mp4', '')
    if camera_key not in camera_params:
        print(f"Параметры камеры для {camera_key} не найдены")
        cap.release()
        return None

    params = camera_params[camera_key]
    resolution = params['video_resolution']
    video_width, video_height = resolution['width'], resolution['height']

    # Извлечение параметров даты
    date_crop = params.get('data_crop_region', {})
    date_digits = params.get('data_digit_coords', {}).get('digits', [])
    date_x_start = int(date_crop.get('x_start', 0) * video_width) if date_crop else 0
    date_y_start = int(date_crop.get('y_start', 0) * video_height) if date_crop else 0
    date_width = int(date_crop.get('width', 0) * video_width) if date_crop else 0
    date_height = int(date_crop.get('height', 0) * video_height) if date_crop else 0
    date_x_start, date_y_start, date_width, date_height = validate_roi(
        date_x_start, date_y_start, date_width, date_height, video_width, video_height
    )
    print(f"Date ROI: x={date_x_start}, y={date_y_start}, w={date_width}, h={date_height}")

    # Извлечение параметров времени
    time_crop = params['time_crop_region']
    time_digits = params['time_digit_coords']['digits']
    time_x_start = int(time_crop['x_start'] * video_width)
    time_y_start = int(time_crop['y_start'] * video_height)
    time_width = int(time_crop['width'] * video_width)
    time_height = int(time_crop['height'] * video_height)
    time_x_start, time_y_start, time_width, time_height = validate_roi(
        time_x_start, time_y_start, time_width, time_height, video_width, video_height
    )
    print(f"Time ROI: x={time_x_start}, y={time_y_start}, w={time_width}, h={time_height}")

    # Преобразуем секунды в номера кадров
    frames_to_check = [int(float(frame_sec) * fps) for frame_sec in metrics_data[video_name]['date'].keys()]
    print(f"Кадры для проверки: {frames_to_check[:10]}...")

    true_data_digits = {frame: metrics_data[video_name]['date'][str(int(frame / fps))] for frame in frames_to_check}
    true_time_digits = {frame: metrics_data[video_name]['time'][str(int(frame / fps))] for frame in frames_to_check}

    # Списки для хранения истинных и предсказанных меток
    data_predictions = {i: {'true': [], 'pred': []} for i in range(8)}
    time_predictions = {i: {'true': [], 'pred': []} for i in range(6)}

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count in frames_to_check:
            # Обработка цифр даты
            if date_crop and date_digits:
                date_roi = frame[date_y_start:date_y_start + date_height, date_x_start:date_x_start + date_width]
                if date_roi.size == 0:
                    print(f"Ошибка: пустой date_roi для кадра {frame_count}")
                    for i in range(8):
                        data_predictions[i]['true'].append(int(true_data_digits[frame_count][str(i)]))
                        data_predictions[i]['pred'].append(-1)  # Пропуск предсказания
                    continue
                for i, digit in enumerate(date_digits):
                    x1 = int(digit['x1'] * date_width)
                    y1 = int(digit['y1'] * date_height)
                    w = int(digit['width'] * date_width)
                    h = int(digit['height'] * date_height)
                    x1, y1, w, h = validate_roi(x1, y1, w, h, date_width, date_height)
                    digit_img = date_roi[y1:y1 + h, x1:x1 + w]
                    if digit_img.size == 0:
                        print(f"Ошибка: пустое изображение для кадра {frame_count}, позиции даты {i}")
                        data_predictions[i]['true'].append(int(true_data_digits[frame_count][str(i)]))
                        data_predictions[i]['pred'].append(-1)
                        continue
                    processed_img = preprocess_image(digit_img, method=method)
                    if processed_img is None:
                        data_predictions[i]['true'].append(int(true_data_digits[frame_count][str(i)]))
                        data_predictions[i]['pred'].append(-1)
                        continue
                    pred_digit = predict_digit(onnx_session, processed_img)
                    true_digit = int(true_data_digits[frame_count][str(i)])
                    data_predictions[i]['true'].append(true_digit)
                    data_predictions[i]['pred'].append(pred_digit)
                    print(f"Дата, кадр {frame_count}, позиция {i}: Истинное {true_digit}, Предсказано {pred_digit}")

            # Обработка цифр времени
            time_roi = frame[time_y_start:time_y_start + time_height, time_x_start:time_x_start + time_width]
            if time_roi.size == 0:
                print(f"Ошибка: пустой time_roi для кадра {frame_count}")
                for i in range(6):
                    time_predictions[i]['true'].append(int(true_time_digits[frame_count][str(i)]))
                    time_predictions[i]['pred'].append(-1)
                continue
            for i, digit in enumerate(time_digits):
                x1 = int(digit['x1'] * time_width)
                y1 = int(digit['y1'] * time_height)
                w = int(digit['width'] * time_width)
                h = int(digit['height'] * time_height)
                x1, y1, w, h = validate_roi(x1, y1, w, h, time_width, time_height)
                digit_img = time_roi[y1:y1 + h, x1:x1 + w]
                if digit_img.size > 0:
                    processed_img = preprocess_image(digit_img, method=method)
                    if processed_img is None:
                        time_predictions[i]['true'].append(int(true_time_digits[frame_count][str(i)]))
                        time_predictions[i]['pred'].append(-1)
                        continue
                    pred_digit = predict_digit(onnx_session, processed_img)
                    true_digit = int(true_time_digits[frame_count][str(i)])
                    time_predictions[i]['true'].append(true_digit)
                    time_predictions[i]['pred'].append(pred_digit)
                    print(f"Время, кадр {frame_count}, позиция {i}: Истинное {true_digit}, Предсказано {pred_digit}")

        frame_count += 1

    cap.release()

    # Вычисление метрик
    metrics = {'data_digits': {}, 'time_digits': {}}
    for i in range(8):
        if data_predictions[i]['true']:
            valid_indices = [idx for idx, pred in enumerate(data_predictions[i]['pred']) if pred != -1]
            true_labels = [data_predictions[i]['true'][idx] for idx in valid_indices]
            pred_labels = [data_predictions[i]['pred'][idx] for idx in valid_indices]
            if true_labels:
                metrics['data_digits'][i] = evaluate_metrics(
                    true_labels, pred_labels, f"Дата, позиция {i+1}"
                )
    for i in range(6):
        if time_predictions[i]['true']:
            valid_indices = [idx for idx, pred in enumerate(time_predictions[i]['pred']) if pred != -1]
            true_labels = [time_predictions[i]['true'][idx] for idx in valid_indices]
            pred_labels = [time_predictions[i]['pred'][idx] for idx in valid_indices]
            if true_labels:
                metrics['time_digits'][i] = evaluate_metrics(
                    true_labels, pred_labels, f"Время, позиция {i+1}"
                )

    # Сохранение метрик в текстовый файл
    output_path = os.path.join(output_dir, f"{video_name}_metrics.txt")
    with open(output_path, 'w', encoding='utf-8') as f:
        for i in range(8):
            if i in metrics['data_digits']:
                f.write(f"Дата, позиция {i+1}, accuracy: {metrics['data_digits'][i]['accuracy']:.4f}\n")
                f.write(f"Дата, позиция {i+1}, f1: {metrics['data_digits'][i]['f1']:.4f}\n")
                f.write(f"Дата, позиция {i+1}, precision: {metrics['data_digits'][i]['precision']:.4f}\n")
                f.write(f"Дата, позиция {i+1}, recall: {metrics['data_digits'][i]['recall']:.4f}\n")
        for i in range(6):
            if i in metrics['time_digits']:
                f.write(f"Время, позиция {i+1}, accuracy: {metrics['time_digits'][i]['accuracy']:.4f}\n")
                f.write(f"Время, позиция {i+1}, f1: {metrics['time_digits'][i]['f1']:.4f}\n")
                f.write(f"Время, позиция {i+1}, precision: {metrics['time_digits'][i]['precision']:.4f}\n")
                f.write(f"Время, позиция {i+1}, recall: {metrics['time_digits'][i]['recall']:.4f}\n")
    print(f"Метрики сохранены в {output_path}")

    return metrics

def main():
    onnx_model_path = "40e_resnet18_classic.onnx"
    video_dir = "sip_12022025/sip_12022025"
    metrics_dir = "metrics_params"
    camera_params_path = "configs/camera_params.json"
    output_dir = "txt_metrics_40e_resnet18_classic"
    method = "Nothing"  # Метод предобработки по умолчанию

    os.makedirs(output_dir, exist_ok=True)

    session = ort.InferenceSession(onnx_model_path)
    input_details = session.get_inputs()
    for input in input_details:
        print(f"Input name: {input.name}, Shape: {input.shape}, Type: {input.type}")

    camera_params = load_json(camera_params_path)

    # Словарь для хранения метрик всех видео
    all_metrics = []

    for metrics_file in os.listdir(metrics_dir):
        if metrics_file.endswith('.json'):
            metrics_path = os.path.join(metrics_dir, metrics_file)
            print(f"Загрузка метрик из: {metrics_path}")
            metrics_data = load_json(metrics_path)
            print(f"Ключи в metrics_data: {metrics_data.keys()}")

            video_name = metrics_file.replace('.json', '.mp4')
            video_path = os.path.join(video_dir, video_name)

            if os.path.exists(video_path):
                print(f"Обработка видео: {video_name}")
                video_metrics = process_video(video_path, camera_params, metrics_data, session, output_dir, method=method)
                if video_metrics:
                    all_metrics.append(video_metrics)
            else:
                print(f"Видео {video_name} не найдено в папке {video_dir}")

    # Вычисление и сохранение средних метрик
    if all_metrics:
        avg_metrics = {'data_digits': {}, 'time_digits': {}}
        # Для даты (8 позиций)
        for i in range(8):
            accuracies = [m['data_digits'][i]['accuracy'] for m in all_metrics if i in m['data_digits']]
            f1_scores = [m['data_digits'][i]['f1'] for m in all_metrics if i in m['data_digits']]
            precisions = [m['data_digits'][i]['precision'] for m in all_metrics if i in m['data_digits']]
            recalls = [m['data_digits'][i]['recall'] for m in all_metrics if i in m['data_digits']]
            if accuracies:
                avg_metrics['data_digits'][i] = {
                    'accuracy': np.mean(accuracies),
                    'f1': np.mean(f1_scores),
                    'precision': np.mean(precisions),
                    'recall': np.mean(recalls)
                }
        # Для времени (6 позиций)
        for i in range(6):
            accuracies = [m['time_digits'][i]['accuracy'] for m in all_metrics if i in m['time_digits']]
            f1_scores = [m['time_digits'][i]['f1'] for m in all_metrics if i in m['time_digits']]
            precisions = [m['time_digits'][i]['precision'] for m in all_metrics if i in m['time_digits']]
            recalls = [m['time_digits'][i]['recall'] for m in all_metrics if i in m['time_digits']]
            if accuracies:
                avg_metrics['time_digits'][i] = {
                    'accuracy': np.mean(accuracies),
                    'f1': np.mean(f1_scores),
                    'precision': np.mean(precisions),
                    'recall': np.mean(recalls)
                }

        # Сохранение средних метрик в текстовый файл
        avg_output_path = os.path.join(output_dir, "average_metrics.txt")
        with open(avg_output_path, 'w', encoding='utf-8') as f:
            for i in range(8):
                if i in avg_metrics['data_digits']:
                    f.write(f"Дата, позиция {i+1}, accuracy: {avg_metrics['data_digits'][i]['accuracy']:.4f}\n")
                    f.write(f"Дата, позиция {i+1}, f1: {avg_metrics['data_digits'][i]['f1']:.4f}\n")
                    f.write(f"Дата, позиция {i+1}, precision: {avg_metrics['data_digits'][i]['precision']:.4f}\n")
                    f.write(f"Дата, позиция {i+1}, recall: {avg_metrics['data_digits'][i]['recall']:.4f}\n")
            for i in range(6):
                if i in avg_metrics['time_digits']:
                    f.write(f"Время, позиция {i+1}, accuracy: {avg_metrics['time_digits'][i]['accuracy']:.4f}\n")
                    f.write(f"Время, позиция {i+1}, f1: {avg_metrics['time_digits'][i]['f1']:.4f}\n")
                    f.write(f"Время, позиция {i+1}, precision: {avg_metrics['time_digits'][i]['precision']:.4f}\n")
                    f.write(f"Время, позиция {i+1}, recall: {avg_metrics['time_digits'][i]['recall']:.4f}\n")
        print(f"Средние метрики сохранены в {avg_output_path}")

if __name__ == "__main__":
    main()
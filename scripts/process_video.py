import cv2
import json
import numpy as np
import onnxruntime as ort
from PIL import Image
import os
from torchvision import transforms

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
    else:
        print(f"Загружены параметры для {base_name}: {params.keys()}")
    return params

def preprocess_image(image, img_size=(32, 32)):
    """
    Предобрабатывает изображение для модели.
    """
    if image.size == 0:
        print("Пустое изображение для предобработки")
        return None
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = Image.fromarray(image)
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5))
    ])
    image = transform(image).unsqueeze(0).numpy()
    return image

def predict_digits(roi_images, session, class_map, expected_digits):
    """
    Предсказывает цифры для списка изображений ROI.
    """
    predictions = []
    for img in roi_images:
        input_data = preprocess_image(img)
        if input_data is None:
            predictions.append('?')
            continue
        outputs = session.run(None, {'input': input_data})[0]
        pred_class = np.argmax(outputs, axis=1)[0]
        predictions.append(class_map.get(pred_class, '?'))

    if len(predictions) == expected_digits:
        return ''.join(predictions)
    else:
        return "Ошибка: неверное количество цифр"

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
    return "Ошибка: неверная длина времени"

def format_date(digits):
    """
    Форматирует строку цифр как дату (DD/MM/YYYY).
    """
    if len(digits) == 8:
        return f"{digits[0:2]}/{digits[2:4]}/{digits[4:8]}"
    return "Ошибка: неверная длина даты"

def process_video(input_video, output_video, onnx_model_path, params_path, img_size=(32, 32)):
    """
    Обрабатывает видео, накладывая предсказанные время и дату.
    """
    # Загрузка параметров камеры
    video_name = os.path.basename(input_video)
    params = load_camera_params(params_path, video_name)
    if not params:
        raise ValueError(f"Параметры для {video_name} не найдены в {params_path}")

    # Загрузка видео
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise ValueError(f"Не удалось открыть видео {input_video}")

    # Параметры видео
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Видео: {video_name}, {width}x{height}, {fps} FPS, {total_frames} кадров")

    # Настройка выходного видео
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    # Загрузка ONNX-модели
    session = ort.InferenceSession(onnx_model_path)

    class_map = {i: str(i) for i in range(10)}
    class_map[10] = ':'  # Если модель поддерживает ':'

    time_crop = params['time_crop_region']
    time_digits = params['time_digit_coords']['digits']
    video_res = params['video_resolution']
    video_width, video_height = video_res['width'], video_res['height']

    # Координаты time_crop в пикселях
    time_x_start = int(time_crop['x_start'] * video_width)
    time_y_start = int(time_crop['y_start'] * video_height)
    time_width = int(time_crop['width'] * video_width)
    time_height = int(time_crop['height'] * video_height)
    time_x_start, time_y_start, time_width, time_height = validate_roi(
        time_x_start, time_y_start, time_width, time_height, video_width, video_height
    )
    print(f"Time ROI: x={time_x_start}, y={time_y_start}, w={time_width}, h={time_height}")

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

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        # print(f"Обработка кадра {frame_count}/{total_frames}")

        # Извлечение ROI времени
        time_roi = frame[time_y_start:time_y_start + time_height, time_x_start:time_x_start + time_width]
        if time_roi.size == 0:
            print("Пустой time_roi")
            time_str = "Ошибка: пустой ROI времени"
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

            digit_string = predict_digits(time_digit_images, session, class_map, expected_digits=6)
            time_str = format_time(digit_string)

        # Извлечение ROI даты
        date_str = "Дата не указана"
        if date_crop and date_digits:
            date_roi = frame[date_y_start:date_y_start + date_height, date_x_start:date_x_start + date_width]
            if date_roi.size == 0:
                print("Пустой date_roi")
                date_str = "Ошибка: пустой ROI даты"
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
                    # else:
                    #     print(f"Пропущена цифра даты {i}: пустое изображение")

                digit_string = predict_digits(date_digit_images, session, class_map, expected_digits=8)
                date_str = format_date(digit_string)

        # Налож supercapacitorение текста
        cv2.putText(
            frame, time_str, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA
        )
        cv2.putText(
            frame, date_str, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA
        )
        out.write(frame)

    # Освобождение ресурсов
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Выходное видео сохранено в {output_video}")

def process_videos(input_folder, output_folder, onnx_model_path, params_path, img_size=(32, 32)):
    """
    Обрабатывает все видео из входной папки и сохраняет результаты в выходную папку.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith('.mp4'):
            input_video = os.path.join(input_folder, filename)
            output_video = os.path.join(output_folder, os.path.splitext(filename)[0] + '_annotated.mp4')
            print(f"Обработка видео: {input_video} -> {output_video}")
            process_video(input_video, output_video, onnx_model_path, params_path, img_size)

if __name__ == "__main__":
    input_folder = r"C:\Users\maria\OneDrive\Documents\job\extract_data\sip_12022025\sip_12022025"
    output_folder = r"C:\Users\maria\OneDrive\Documents\job\extract_data\demo\40e_resnet18_classic"
    onnx_model_path = r"C:\Users\maria\OneDrive\Documents\job\extract_data\project\compare\40e_resnet18_classic.onnx"
    params_path = r"C:\Users\maria\OneDrive\Documents\job\extract_data\configs\camera_params.json"

    process_videos(input_folder, output_folder, onnx_model_path, params_path)
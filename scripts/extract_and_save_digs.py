import cv2
import json
import os
import glob

def extract_and_save_individual_chars(video_path, config_data, output_base_dir="verified_cropped_chars"):
    """
    Извлекает и сохраняет изображения отдельных символов (цифр и разделителей)
    на основе точных координат, сохраненных в JSON-файле, с обработкой нескольких кадров.
    """
    video_filename_base = os.path.splitext(os.path.basename(video_path))[0]
    camera_config = config_data.get(video_filename_base)

    if not camera_config:
        print(f"Конфигурация для видео '{video_filename_base}' не найдена. Пропускаем.")
        return

    # Загружаем видео
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Не удалось открыть видеофайл: {video_path}. Пропускаем.")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Извлекаем кадры через равные интервалы (например, каждые 5% длины видео)
    num_intervals = 100  # Количество интервалов (20 кадров = 5% каждый)
    frame_indices = [int(total_frames * i / num_intervals) for i in range(num_intervals)]

    output_video_dir = os.path.join(output_base_dir, video_filename_base)
    os.makedirs(output_video_dir, exist_ok=True)

    # --- Функция для обработки ROI и извлечения отдельных символов ---
    def process_and_extract_chars(frame_image, frame_idx, region_type, crop_region_params, digit_coords_data):
        """
        Обрезает основную ROI, а затем извлекает отдельные символы по их сохраненным координатам.
        """
        x_start_roi_rel = crop_region_params["x_start"]
        y_start_roi_rel = crop_region_params["y_start"]
        width_roi_rel = crop_region_params["width"]
        height_roi_rel = crop_region_params["height"]

        x1_roi_px = int(x_start_roi_rel * video_width)
        y1_roi_px = int(y_start_roi_rel * video_height)
        w_roi_px = int(width_roi_rel * video_width)
        h_roi_px = int(height_roi_rel * video_height)

        x1_roi_px = max(0, x1_roi_px)
        y1_roi_px = max(0, y1_roi_px)
        w_roi_px = min(w_roi_px, video_width - x1_roi_px)
        h_roi_px = min(h_roi_px, video_height - y1_roi_px)

        roi_image = frame_image[y1_roi_px : y1_roi_px + h_roi_px, x1_roi_px : x1_roi_px + w_roi_px]

        digit_coords_list = digit_coords_data.get("digits", [])
        num_expected_digits = digit_coords_data.get("num_digits", 0)

        print(f"  Обработка {region_type} (кадр {frame_idx}, {len(digit_coords_list)} из {num_expected_digits} символов)...")
        for j, digit_coord_rel in enumerate(digit_coords_list):
            x1_digit_rel = digit_coord_rel["x1"]
            y1_digit_rel = digit_coord_rel["y1"]
            width_digit_rel = digit_coord_rel["width"]
            height_digit_rel = digit_coord_rel["height"]

            x1_digit_px = int(x1_digit_rel * w_roi_px)
            y1_digit_px = int(y1_digit_rel * h_roi_px)
            w_digit_px = int(width_digit_rel * w_roi_px)
            h_digit_px = int(height_digit_rel * h_roi_px)
            
            x1_digit_px = max(0, x1_digit_px)
            y1_digit_px = max(0, y1_digit_px)
            w_digit_px = min(w_digit_px, roi_image.shape[1] - x1_digit_px)
            h_digit_px = min(h_digit_px, roi_image.shape[0] - y1_digit_px)

            char_img = roi_image[y1_digit_px : y1_digit_px + h_digit_px, x1_digit_px : x1_digit_px + w_digit_px]
            output_path = os.path.join(output_video_dir, f"{region_type}_char_frame_{frame_idx}_pos_{j}.png")
            cv2.imwrite(output_path, char_img)

    # Обрабатываем каждый выбранный кадр
    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            print(f"Не удалось считать кадр {frame_idx} из видео: {video_path}. Пропускаем.")
            continue

        # Обработка даты
        date_crop_region = camera_config.get("data_crop_region")
        date_digit_coords_data = camera_config.get("data_digit_coords")
        process_and_extract_chars(frame, frame_idx, "date", date_crop_region, date_digit_coords_data)

        # Обработка времени
        time_crop_region = camera_config.get("time_crop_region")
        time_digit_coords_data = camera_config.get("time_digit_coords")
        process_and_extract_chars(frame, frame_idx, "time", time_crop_region, time_digit_coords_data)

    cap.release()

config_file = "configs/camera_params.json"
video_source_folder = "sip_12022025/sip_12022025"
output_char_verification_folder = "cropped_imgs"

with open(config_file, 'r') as f:
    all_camera_configs = json.load(f)

os.makedirs(output_char_verification_folder, exist_ok=True)

# Получаем список всех видеофайлов в указанной папке
video_files = glob.glob(os.path.join(video_source_folder, "*.mp4"))

for video_path in video_files:
    print(f"\nОбработка видео: {os.path.basename(video_path)}")
    extract_and_save_individual_chars(video_path, all_camera_configs, output_char_verification_folder)

print("DONE")
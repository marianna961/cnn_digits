import cv2
import json
import os
import glob

# Коэффициент масштабирования для окна выделения цифр
scale_factor = 4

# main ROI
def select_roi_callback(event, x, y, flags, param):
    data = param['data']

    if event == cv2.EVENT_LBUTTONDOWN:
        if not data['drawing']:
            data['p1'] = (x, y)
            data['drawing'] = True
    elif event == cv2.EVENT_LBUTTONUP:
        if data['drawing']:
            data['p2'] = (x, y)
            data['drawing'] = False
            
            x1, y1 = min(data['p1'][0], data['p2'][0]), min(data['p1'][1], data['p2'][1])
            x2, y2 = max(data['p1'][0], data['p2'][0]), max(data['p1'][1], data['p2'][1])

            data['roi_coords'] = ((x1, y1), (x2, y2))
            
            # copy
            temp_img = data['img_for_display'].copy()
            cv2.rectangle(temp_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.imshow("Select ROI", temp_img)
    elif event == cv2.EVENT_MOUSEMOVE:
        if data['drawing']:
            temp_img = data['img_for_display'].copy()
            x1_draw, y1_draw = data['p1']
            cv2.rectangle(temp_img, (x1_draw, y1_draw), (x, y), (0, 255, 0), 2)
            cv2.imshow("Select ROI", temp_img)


# digit
def select_digit_callback(event, x, y, flags, param):
    data = param['data']
    
    if event == cv2.EVENT_LBUTTONDOWN:
        if not data['drawing']:
            orig_x, orig_y = int(x / scale_factor), int(y / scale_factor)
            data['digit_coords'].append([(orig_x, orig_y)])
            data['drawing'] = True
    elif event == cv2.EVENT_LBUTTONUP:
        if data['drawing']:
            orig_x, orig_y = int(x / scale_factor), int(y / scale_factor)
            data['digit_coords'][data['current_digit_index']].append((orig_x, orig_y))
            data['drawing'] = False
            
            p1 = data['digit_coords'][data['current_digit_index']][0]
            p2 = data['digit_coords'][data['current_digit_index']][1]
            x1, y1 = min(p1[0], p2[0]), min(p1[1], p2[1])
            x2, y2 = max(p1[0], p2[0]), max(p1[1], p2[1])
            
            temp_scaled_roi_img = data['scaled_roi_img'].copy()
            for i, d_coords in enumerate(data['digit_coords']):
                if len(d_coords) == 2:
                    px1, py1 = d_coords[0]
                    px2, py2 = d_coords[1]
                    draw_x1, draw_y1 = int(min(px1, px2) * scale_factor), int(min(py1, py2) * scale_factor)
                    draw_x2, draw_y2 = int(max(px1, px2) * scale_factor), int(max(py1, py2) * scale_factor)
                    cv2.rectangle(temp_scaled_roi_img, (draw_x1, draw_y1), (draw_x2, draw_y2), (0, 255, 0), 2)
            cv2.imshow("select digits in ROI", temp_scaled_roi_img)
            
            data['current_digit_index'] += 1
    elif event == cv2.EVENT_MOUSEMOVE:
        if data['drawing']:
            temp_scaled_roi_img = data['scaled_roi_img'].copy()
            for i, d_coords in enumerate(data['digit_coords']):
                if len(d_coords) == 2:
                    px1, py1 = d_coords[0]
                    px2, py2 = d_coords[1]
                    draw_x1, draw_y1 = int(min(px1, px2) * scale_factor), int(min(py1, py2) * scale_factor)
                    draw_x2, draw_y2 = int(max(px1, px2) * scale_factor), int(max(py1, py2) * scale_factor)
                    cv2.rectangle(temp_scaled_roi_img, (draw_x1, draw_y1), (draw_x2, draw_y2), (0, 255, 0), 2)
            
            p1_draw = data['digit_coords'][data['current_digit_index']][0]
            draw_x1_curr, draw_y1_curr = int(p1_draw[0] * scale_factor), int(p1_draw[1] * scale_factor)
            draw_x2_curr, draw_y2_curr = x, y
            cv2.rectangle(temp_scaled_roi_img, (draw_x1_curr, draw_y1_curr), (draw_x2_curr, draw_y2_curr), (0, 255, 0), 2)
            cv2.imshow("select digits in ROI", temp_scaled_roi_img)


video_folder = "sip_12022025/sip_12022025"
output_config_dir = "configs"
output_config_file = os.path.join(output_config_dir, "camera_params.json")

# Загружаем существующие конфигурации, если файл существует
all_camera_configs = {}
if os.path.exists(output_config_file):
    with open(output_config_file, 'r') as f:
        all_camera_configs = json.load(f)


os.makedirs(output_config_dir, exist_ok=True)

video_files = glob.glob(os.path.join(video_folder, "*.mp4"))


for i, video_path in enumerate(video_files):
    video_filename_base = os.path.splitext(os.path.basename(video_path))[0]
    print(f"{i+1} из {len(video_files)}: {video_filename_base}")

    cap = cv2.VideoCapture(video_path)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_to_open = int(total_frames * 0.10) # 10% от начала
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_to_open)

    ret, img = cap.read()
    cap.release()

    video_width = img.shape[1]
    video_height = img.shape[0]

    camera_config = {
        "video_resolution": {"width": video_width, "height": video_height}
    }
    
    # ROI для даты
    date_roi_selection_data = {
        'p1': None, 'p2': None, 'drawing': False, 'roi_coords': None, 'img_for_display': img.copy()
    }
    cv2.namedWindow("select ROI")
    cv2.setMouseCallback("select ROI", select_roi_callback, param={'data': date_roi_selection_data})

    print("ROI для ДАТЫ")
    cv2.imshow("select ROI", date_roi_selection_data['img_for_display'])
    k = cv2.waitKey(0) & 0xFF
    cv2.destroyWindow("Select ROI")

    if k == 13 and date_roi_selection_data['roi_coords']: # Enter и ROI выделена
        (x1, y1), (x2, y2) = date_roi_selection_data['roi_coords']
        camera_config["data_crop_region"] = {
            "x_start": round(x1 / video_width, 4),
            "y_start": round(y1 / video_height, 4),
            "width": round((x2 - x1) / video_width, 4),
            "height": round((y2 - y1) / video_height, 4)
        }
        
        # обрезаем ROI даты для выделения цифр
        roi_date_x1 = int(camera_config["data_crop_region"]["x_start"] * video_width)
        roi_date_y1 = int(camera_config["data_crop_region"]["y_start"] * video_height)
        roi_date_width = int(camera_config["data_crop_region"]["width"] * video_width)
        roi_date_height = int(camera_config["data_crop_region"]["height"] * video_height)

        roi_img_date = img[roi_date_y1 : roi_date_y1 + roi_date_height, 
                           roi_date_x1 : roi_date_x1 + roi_date_width]
        scaled_roi_img_date = cv2.resize(roi_img_date, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)

        #  цифры для даты
        digit_selection_data_date = {
            'drawing': False, 
            'digit_coords': [], 
            'current_digit_index': 0, 
            'scaled_roi_img': scaled_roi_img_date
        }
        cv2.namedWindow("select digits in date ROI")
        cv2.setMouseCallback("select digits in date ROI", select_digit_callback, param={'data': digit_selection_data_date})
        
        print("======ROI для Даты======")
        
        while True:
            temp_scaled_roi_img = digit_selection_data_date['scaled_roi_img'].copy()
            # отрисовка прямоугольников
            for d_coords in digit_selection_data_date['digit_coords']:
                if len(d_coords) == 2:
                    px1, py1 = d_coords[0]
                    px2, py2 = d_coords[1]
                    draw_x1, draw_y1 = int(min(px1, px2) * scale_factor), int(min(py1, py2) * scale_factor)
                    draw_x2, draw_y2 = int(max(px1, px2) * scale_factor), int(max(py1, py2) * scale_factor)
                    cv2.rectangle(temp_scaled_roi_img, (draw_x1, draw_y1), (draw_x2, draw_y2), (0, 255, 0), 2)
            cv2.imshow("select digits in date ROI", temp_scaled_roi_img)
            k = cv2.waitKey(1) & 0xFF
            if k == 13: # Enter
                break
            elif k == ord('c'):
                digit_selection_data_date['digit_coords'] = []
                break

        cv2.destroyWindow("select digits in date ROI")

        # координаты цифр даты
        data_digit_info = []
        for d_coords in digit_selection_data_date['digit_coords']:
            p1_orig = d_coords[0]
            p2_orig = d_coords[1]
            
            x1_px = min(p1_orig[0], p2_orig[0])
            y1_px = min(p1_orig[1], p2_orig[1])
            x2_px = max(p1_orig[0], p2_orig[0])
            y2_px = max(p1_orig[1], p2_orig[1])

            relative_x1 = x1_px / roi_img_date.shape[1]
            relative_y1 = y1_px / roi_img_date.shape[0]
            relative_width = (x2_px - x1_px) / roi_img_date.shape[1]
            relative_height = (y2_px - y1_px) / roi_img_date.shape[0]

            data_digit_info.append({
                "x1": round(relative_x1, 4),
                "y1": round(relative_y1, 4),
                "width": round(relative_width, 4),
                "height": round(relative_height, 4)
            })
        
        camera_config["data_digit_coords"] = {
            "num_digits": len(data_digit_info),
            "digits": data_digit_info
        }
        # параметры сегментации 
        if len(data_digit_info) > 1:
            total_width_sum = sum(d["width"] for d in data_digit_info)
            avg_width = total_width_sum / len(data_digit_info)
            
            total_spacing_sum = 0
            for k in range(len(data_digit_info) - 1):
                total_spacing_sum += data_digit_info[k+1]["x1"] - (data_digit_info[k]["x1"] + data_digit_info[k]["width"])
            avg_spacing = total_spacing_sum / (len(data_digit_info) - 1) if len(data_digit_info) > 1 else 0

            # padding_estimation
            padding_est = data_digit_info[0]["x1"] # Расстояние от левого края ROI до первого символа
            if len(data_digit_info) > 0:
                 padding_est = data_digit_info[0]["x1"] 
                 
            camera_config["data_digit_segmentation_params"] = {
                "num_digits": len(data_digit_info),
                "average_digit_width": round(avg_width, 4),
                "inter_digit_spacing": round(avg_spacing, 4),
                "padding": round(padding_est, 4)
            }


    elif k == ord('c'):
        print("пропуск")

    
    # ROI для времени
    time_roi_selection_data = {
        'p1': None, 'p2': None, 'drawing': False, 'roi_coords': None, 'img_for_display': img.copy()
    }
    cv2.namedWindow("Select ROI")
    cv2.setMouseCallback("Select ROI", select_roi_callback, param={'data': time_roi_selection_data})

    print("ROI для времени")
    cv2.imshow("Select ROI", time_roi_selection_data['img_for_display'])
    k = cv2.waitKey(0) & 0xFF
    cv2.destroyWindow("Select ROI")

    if k == 13 and time_roi_selection_data['roi_coords']: # Enter и ROI выделена
        (x1, y1), (x2, y2) = time_roi_selection_data['roi_coords']
        camera_config["time_crop_region"] = {
            "x_start": round(x1 / video_width, 4),
            "y_start": round(y1 / video_height, 4),
            "width": round((x2 - x1) / video_width, 4),
            "height": round((y2 - y1) / video_height, 4)
        }

        # Обрезаем ROI времени для выделения цифр
        roi_time_x1 = int(camera_config["time_crop_region"]["x_start"] * video_width)
        roi_time_y1 = int(camera_config["time_crop_region"]["y_start"] * video_height)
        roi_time_width = int(camera_config["time_crop_region"]["width"] * video_width)
        roi_time_height = int(camera_config["time_crop_region"]["height"] * video_height)

        roi_img_time = img[roi_time_y1 : roi_time_y1 + roi_time_height, 
                            roi_time_x1 : roi_time_x1 + roi_time_width]
        scaled_roi_img_time = cv2.resize(roi_img_time, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)

        # цифры для времени
        digit_selection_data_time = {
            'drawing': False, 
            'digit_coords': [], 
            'current_digit_index': 0, 
            'scaled_roi_img': scaled_roi_img_time
        }
        cv2.namedWindow("select digits in time ROI")
        cv2.setMouseCallback("select digits in time ROI", select_digit_callback, param={'data': digit_selection_data_time})
        
        print("ROI для цифр времени")
        
        while True:
            temp_scaled_roi_img = digit_selection_data_time['scaled_roi_img'].copy()
            for d_coords in digit_selection_data_time['digit_coords']:
                if len(d_coords) == 2:
                    px1, py1 = d_coords[0]
                    px2, py2 = d_coords[1]
                    draw_x1, draw_y1 = int(min(px1, px2) * scale_factor), int(min(py1, py2) * scale_factor)
                    draw_x2, draw_y2 = int(max(px1, px2) * scale_factor), int(max(py1, py2) * scale_factor)
                    cv2.rectangle(temp_scaled_roi_img, (draw_x1, draw_y1), (draw_x2, draw_y2), (0, 255, 0), 2)
            cv2.imshow("select digits in time ROI", temp_scaled_roi_img)
            k = cv2.waitKey(1) & 0xFF
            if k == 13: #Enter
                break
            elif k == ord('c'):
                digit_selection_data_time['digit_coords'] = []
                break

        cv2.destroyWindow("select digits in time ROI")

        # координаты цифр времени
        time_digit_info = []
        for d_coords in digit_selection_data_time['digit_coords']:
            p1_orig = d_coords[0]
            p2_orig = d_coords[1]
            
            x1_px = min(p1_orig[0], p2_orig[0])
            y1_px = min(p1_orig[1], p2_orig[1])
            x2_px = max(p1_orig[0], p2_orig[0])
            y2_px = max(p1_orig[1], p2_orig[1])

            relative_x1 = x1_px / roi_img_time.shape[1] # относительно ширины ROI времени
            relative_y1 = y1_px / roi_img_time.shape[0] # относительно высоты ROI времени
            relative_width = (x2_px - x1_px) / roi_img_time.shape[1]
            relative_height = (y2_px - y1_px) / roi_img_time.shape[0]

            time_digit_info.append({
                "x1": round(relative_x1, 4),
                "y1": round(relative_y1, 4),
                "width": round(relative_width, 4),
                "height": round(relative_height, 4)
            })
        
        camera_config["time_digit_coords"] = {
            "num_digits": len(time_digit_info),
            "digits": time_digit_info
        }

        if len(time_digit_info) > 1:
            total_width_sum = sum(d["width"] for d in time_digit_info)
            avg_width = total_width_sum / len(time_digit_info)
            
            total_spacing_sum = 0
            for k in range(len(time_digit_info) - 1):
                total_spacing_sum += time_digit_info[k+1]["x1"] - (time_digit_info[k]["x1"] + time_digit_info[k]["width"])
            avg_spacing = total_spacing_sum / (len(time_digit_info) - 1) if len(time_digit_info) > 1 else 0

            padding_est = time_digit_info[0]["x1"] 

            camera_config["time_digit_segmentation_params"] = {
                "num_digits": len(time_digit_info),
                "average_digit_width": round(avg_width, 4),
                "inter_digit_spacing": round(avg_spacing, 4),
                "padding": round(padding_est, 4)
            }

    elif k == ord('c'):
        print("пропуск")

    all_camera_configs[video_filename_base] = camera_config
    with open(output_config_file, 'w') as f:
        json.dump(all_camera_configs, f, indent=4)

cv2.destroyAllWindows()
print("DONE")
import cv2
import numpy as np
import streamlit as st
import openvino as ov
import os

# 모델 파일 경로 설정
model_dir = 'models'
car_model_path = os.path.join(model_dir, 'vehicle-detection-0200.xml')  # 실제 모델 파일 경로로 업데이트하세요

# 모델 파일이 존재하는지 확인
if not os.path.exists(car_model_path):
    raise FileNotFoundError(f"Model file not found: {car_model_path}")

# OpenVINO Core 객체 생성
core = ov.Core()

# 모델 로드 및 컴파일
model_car = core.read_model(model=car_model_path)
compiled_model_car = core.compile_model(model=model_car, device_name="CPU")

input_layer_car = compiled_model_car.input(0)
output_layer_car = compiled_model_car.output(0)

def preprocess(frame, input_layer_car):
    """
    이미지를 모델의 입력 형식으로 전처리합니다.
    """
    N, input_channels, input_height, input_width = input_layer_car.shape

    if frame is None:
        return None

    resized_frame = cv2.resize(frame, (input_width, input_height))
    transposed_frame = resized_frame.transpose(2, 0, 1)
    input_frame = np.expand_dims(transposed_frame, 0)

    return input_frame

def find_car_boxes(frame, results, confidence_threshold):
    """
    모델의 결과에서 차량의 박스와 신뢰도 점수를 추출합니다.
    """
    results = results.squeeze()

    scores = results[:, 2]
    boxes = results[:, -4:]

    car_boxes = boxes[scores >= confidence_threshold]
    scores = scores[scores >= confidence_threshold]

    frame_h, frame_w, frame_channels = frame.shape

    car_boxes = car_boxes * np.array([frame_w, frame_h, frame_w, frame_h])
    car_boxes = car_boxes.astype(np.int64)

    return car_boxes, scores

def draw_car_boxes(car_boxes, frame):
    """
    차량의 박스를 이미지에 그립니다.
    """
    show_frame = frame.copy()

    for i in range(len(car_boxes)):
        xmin, ymin, xmax, ymax = car_boxes[i]
        car = frame[ymin:ymax, xmin:xmax]

        if car.size == 0:
            continue

        # --- Drawing ---
        fontScale = frame.shape[1] / 750
        text = f"Car"
        
        box_color = (0, 0, 255)  # 차량을 위한 빨간색
        cv2.putText(show_frame, text, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, fontScale, (0, 0, 255), 2)
        cv2.rectangle(show_frame, (xmin, ymin), (xmax, ymax), box_color, 2)

    # 차량 수에 따라 무게와 상태 표시
    m = 1600 * len(car_boxes)
    
    if len(car_boxes) >= 7:
        traffic_text = 'Excessive load'
        st.toast('Excessive load')
    else:
        traffic_text = 'normal load'
        

    num_cars_text = f"Detected cars: {len(car_boxes)} {traffic_text} {m}kg"
    cv2.putText(show_frame, num_cars_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    return show_frame, len(car_boxes)

def predict_image(image, conf_threshold):
    """
    이미지를 입력으로 받아 차량을 감지하고 결과를 반환합니다.
    """
    input_image = preprocess(image, input_layer_car)
    results = compiled_model_car([input_image])[output_layer_car]
    car_boxes, scores = find_car_boxes(image, results, conf_threshold)
    visualize_image, num_cars = draw_car_boxes(car_boxes, image)
    
    return visualize_image, num_cars

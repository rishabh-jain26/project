import cv2
import numpy as np
import tensorflow as tf

def detect_vehicles(image, model, image_height, image_width):
    height, width, channels = image.shape
    resized_image = cv2.resize(image, (image_height, image_width))
    normalized_image = resized_image / 255.0
    input_data = np.expand_dims(normalized_image, axis=0)
    predictions = model.predict(input_data)[0]
    vehicles = []
    for i in range(len(predictions)):
        if predictions[i] > 0.5:
            x, y, w, h = labels[i]
            vehicles.append((x, y, w, h))
    return vehicles

def is_cutting_in(prev_vehicle, curr_vehicle, threshold=30):
    px, py, pw, ph = prev_vehicle
    cx, cy, cw, ch = curr_vehicle
    if abs(cx - px) > threshold:
        return True
    return False

def detect_cut_in(vehicles, frame_sequence, threshold=30):
    cut_in_events = []
    for i in range(1, len(frame_sequence)):
        prev_frame_vehicles = vehicles[i - 1]
        curr_frame_vehicles = vehicles[i]
        for pv in prev_frame_vehicles:
            for cv in curr_frame_vehicles:
                if is_cutting_in(pv, cv, threshold):
                    cut_in_events.append((i, pv, cv))
    return cut_in_events

model = tf.keras.models.load_model('vehicle_cut_in_detection_model.h5')

image_height, image_width = 416, 416 

video_path = 'car.mp4'
cap = cv2.VideoCapture(video_path)
frames = []
while(cap.isOpened()):
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(frame)

vehicles_in_frames = []
for frame in frames:
    vehicles = detect_vehicles(frame, model, image_height, image_width)
    vehicles_in_frames.append(vehicles)

cut_in_events = detect_cut_in(vehicles_in_frames, frames)

for event in cut_in_events:
    frame_index, prev_vehicle, curr_vehicle = event
    frame = frames[frame_index]
    px, py, pw, ph = prev_vehicle
    cx, cy, cw, ch = curr_vehicle
    color = (0, 0, 255)
    cv2.rectangle(frame, (px, py), (px + pw, py + ph), color, 2)
    cv2.rectangle(frame, (cx, cy), (cx + cw, cy + ch), color, 2)
    cv2.imwrite(f"cut_in_event_{frame_index}.jpg", frame)
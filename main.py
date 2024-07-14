import cv2
from cut_detect import detect_cut_in

def main(video_path, model_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    model = tf.keras.models.load_model(model_path)

    image_height, image_width = 416, 416 

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

if __name__ == "_main_":
    video_path = 'car.mp4'
    model_path = 'vehicle_cut_in_detection_model.h5'
    main(video_path, model_path)
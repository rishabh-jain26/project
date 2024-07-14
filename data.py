import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

def load_idd_data(data_dir):
    images = []
    labels = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".jpg"):
                img_path = os.path.join(root, file)
                label_path = img_path.replace(".jpg", ".txt")
                image = cv2.imread(img_path)
                height, width, _ = image.shape
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        class_id = int(parts[0])
                        if class_id in [0, 1, 2, 3]:
                            x_center = float(parts[1]) * width
                            y_center = float(parts[2]) * height
                            w = float(parts[3]) * width
                            h = float(parts[4]) * height
                            x = int(x_center - w / 2)
                            y = int(y_center - h / 2)
                            images.append(image)
                            labels.append((x, y, w, h))
    return images, labels

data_dir = 'Indian_driving_dataset.csv'
images, labels = load_idd_data(data_dir)
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

np.save('X_train.npy', X_train)
np.save('X_test.npy', X_test)
np.save('y_train.npy', y_train)
np.save('y_test.npy', y_test)
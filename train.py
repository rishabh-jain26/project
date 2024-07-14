import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

X_train = np.load('X_train.npy')
X_test = np.load('X_test.npy')
y_train = np.load('y_train.npy')
y_test = np.load('y_test.npy')

image_height, image_width = X_train[0].shape[0], X_train[0].shape[1]
channels = X_train[0].shape[2]
num_classes = 4

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(image_height, image_width, channels)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

y_train = np.array([label[0] for label in y_train])
y_test = np.array([label[0] for label in y_test])

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)
datagen.fit(X_train)


history = model.fit(datagen.flow(np.array(X_train), y_train, batch_size=32),
                    epochs=10,
                    validation_data=(np.array(X_test), y_test))


test_loss, test_acc = model.evaluate(np.array(X_test), y_test, verbose=2)
print(f'\nTest accuracy: {test_acc}')

model.save('vehicle_cut_in_detection_model.h5')
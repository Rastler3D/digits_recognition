import os
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from collections import defaultdict

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))

train_images = train_images.astype('float32') / 255
test_images = test_images.astype('float32') / 255

train_labels = keras.utils.to_categorical(train_labels)
test_labels = keras.utils.to_categorical(test_labels)

model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=5, batch_size=64, validation_split=0.2)

test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f'Точность на тестовых данных: {test_acc}')

def preprocess_image(image_path):
    img = load_img(image_path, color_mode='grayscale', target_size=(28, 28))
    img_array = img_to_array(img)
    img_array = img_array.reshape((1, 28, 28, 1))
    img_array = img_array.astype('float32') / 255
    return img_array

def analyze_images_in_folder(folder_path):
    results = defaultdict(int)
    i = 0
    for filename in os.listdir(folder_path):
        if i > 100:
            break
        i+=1
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(folder_path, filename)
            processed_image = preprocess_image(image_path)
            prediction = model.predict(processed_image)
            predicted_digit = np.argmax(prediction)
            results[predicted_digit] += 1
            print(f'Файл: {filename} - Предсказанная цифра: {predicted_digit}')
    return dict(results)

folder_path = './digits'
statistics = analyze_images_in_folder(folder_path)

result = [v for _, v in sorted(statistics.items())]
print(result)
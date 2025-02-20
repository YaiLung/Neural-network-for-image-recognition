import os
import numpy as np
from tensorflow.keras import layers
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image, ImageDraw
import io
import sys

# Проверяем, доступен ли tkinter (нужен только для GUI)
try:
    import tkinter as tk
    from tkinter import messagebox
    GUI_AVAILABLE = True
except ImportError:
    GUI_AVAILABLE = False

# Константы
SIZE = 28
CHEERFUL_COLOR = (255, 255, 255)
SAD_COLOR = (255, 255, 255)

# Создаем датасет смайликов
cheerful_smileys = []
sad_smileys = []

for i in range(1000):
    cheerful_smiley = Image.new('RGB', (SIZE, SIZE), color=CHEERFUL_COLOR)
    draw = ImageDraw.Draw(cheerful_smiley)
    draw.arc((3, 3, SIZE-4, SIZE-3), 0, 180, fill=(0, 0, 0), width=2)
    draw.rectangle((6, 3, 8, 7), fill=(0, 0, 0))
    draw.rectangle((20, 3, 22, 7), fill=(0, 0, 0))
    cheerful_smileys.append(np.array(cheerful_smiley))

    sad_smiley = Image.new('RGB', (SIZE, SIZE), color=SAD_COLOR)
    draw = ImageDraw.Draw(sad_smiley)
    draw.arc((3, 12, SIZE-4, SIZE+10), 180, 360, fill=(0, 0, 0), width=2)
    draw.rectangle((6, 3, 8, 7), fill=(0, 0, 0))
    draw.rectangle((20, 3, 22, 7), fill=(0, 0, 0))
    sad_smileys.append(np.array(sad_smiley))

# Данные для модели
images = np.array(cheerful_smileys + sad_smileys)
labels = np.array([1] * 1000 + [0] * 1000)
train_images, test_images = images[:1600] / 255.0, images[1600:] / 255.0
train_labels, test_labels = labels[:1600], labels[1600:]

# Аугментация данных
train_datagen = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=True
)
train_datagen.fit(train_images)
train_generator = train_datagen.flow(train_images, train_labels, batch_size=10)

# Создание модели CNN
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(SIZE, SIZE, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(2, activation='softmax')
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Обучение модели
model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))

# Запуск GUI только в локальной среде
if GUI_AVAILABLE and __name__ == "__main__":
    root = tk.Tk()
    root.title("DRAW SMILE")

    canvas_width = 56 * 10
    canvas_height = 56 * 10
    canvas = tk.Canvas(root, width=canvas_width, height=canvas_height, bg='white')
    canvas.pack()

    def paint(event):
        x1, y1 = (event.x - 10), (event.y - 10)
        x2, y2 = (event.x + 10), (event.y + 10)
        canvas.create_oval(x1, y1, x2, y2, fill="black")

    def predict_smiley():
        # Создаем изображение с холста
        ps = canvas.postscript(colormode='color')
        img = Image.open(io.BytesIO(ps.encode('utf-8')))
        img = img.convert('RGB').resize((SIZE, SIZE))
        img = np.array(img) / 255.0
        img = img.reshape(1, SIZE, SIZE, 3)

        # Предсказание
        pred = model.predict(img)
        pred_class = np.argmax(pred)

        # Вывод результата
        messagebox.showinfo("Prediction", "Cheerful!" if pred_class == 1 else "Sad!")

    def clear_canvas():
        canvas.delete("all")

    canvas.bind("<B1-Motion>", paint)
    tk.Button(root, text="Check", command=predict_smiley).pack()
    tk.Button(root, text="Clear", command=clear_canvas).pack()

    root.mainloop()





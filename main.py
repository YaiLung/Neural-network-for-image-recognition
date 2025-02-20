import numpy as np
from tensorflow.keras import layers
from tkinter import *
from PIL import ImageDraw
from tensorflow import keras
from tkinter import messagebox
from PIL import Image
import io
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# Define the colors of the smileys
CHEERFUL_COLOR = (255, 255, 255)  # yellow
SAD_COLOR = (255, 255, 255)  # blue
AVG_COLOR = (127, 127, 127)
# Define the size of the smileys
SIZE = 28

# Create the dataset of smiley images
cheerful_smileys = []
sad_smileys = []

# Define the data augmentation transformations



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
    draw.rectangle((8, 7, 6, 3), fill=(0, 0, 0))
    draw.rectangle((22, 7, 20, 3), fill=(0, 0, 0))
    sad_smileys.append(np.array(sad_smiley))

# сэт + хэппи
images = np.array(cheerful_smileys + sad_smileys)
labels = np.array([1]*1000 + [0]*1000)

# Разделите данные на обучающие и тестовые наборы 1600 обучающихся и 400 тестовх
train_images, test_images = images[:1600] / 255.0, images[1600:] / 255.0
train_labels, test_labels = labels[:1600], labels[1600:]
# Define the data augmentation transformations
train_datagen = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=True
)


# Fit the data augmentation generator to the training data
train_datagen.fit(train_images)

# Create a generator to yield augmented training data batches
train_generator = train_datagen.flow(train_images, train_labels, batch_size=10)

#Мы определим простую архитектуру CNN с двумя сверточными слоями
#за которыми следуют два полностью связанных слоя. Мы будем использовать функцию активации
#ReLU для скрытых слоев и функцию активации softmax для выходного слоя.

#Мы используем последовательную модель от Keras для создания архитектуры CNN.
#Мы начинаем с Conv2DLayer с 32 фильтрами размером 3x3, за которым следует слой MaxPooling2D с размером пула 2x2
# Мы повторяем это с другим Conv2DLayer с 64 фильтрами размером 3x3, за которым следует другой слой MaxPooling2D с размером пула 2x2.
#Затем мы сглаживаем выходные данные и пропускаем их через два полностью соединенных слоя с функциями активации ReLU.
#Наконец, у нас есть выходной слой с 2 нейронами и функцией активации softmax, которая дает нам вероятности каждого класса (грустный или веселый).
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(SIZE, SIZE, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(2, activation='softmax')
])

#мы будем использовать оптимизатор Adam и категориальную функцию потери перекрестной энтропии
#Мы также будем следить за точностью модели во время обучения.
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# тренируем модель 10 эпох
model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels,))

# Теперь мы создадим графический пользовательский интерфейс с помощью tkinter
# GUI = graphical user interface (графический пользовательский интерфейс)
root = Tk()
root.title("DRAW SMILE")

canvas_width = 56 * 10
canvas_height = 56 * 10
canvas = Canvas(root, width=canvas_width, height=canvas_height, bg='white')
canvas.pack()

def paint(event):
    x1, y1 = (event.x - 10), (event.y - 10)
    x2, y2 = (event.x + 10), (event.y + 10)
    canvas.create_oval(x1, y1, x2, y2, fill="black")
#Теперь мы определим функцию predict_smiley, которая принимает нарисованное изображение смайлика,
#предварительно обрабатывает его и передает через обученную модель, чтобы получить предсказанный класс.
def predict_smiley():
    # Resize the image to the required size
    img = canvas.postscript(colormode='mono')
    img = Image.open(io.BytesIO(img.encode('utf-8')))
    img = img.resize((SIZE, SIZE))
    img = np.array(img)

    #Сначала мы изменяем размер нарисованного смайлика до требуемого размера,
    #а затем предварительно обрабатываем его, изменяя его форму и нормализуя значения пикселей.
    img = img.reshape(1, SIZE, SIZE, 3) / 255.0

    # Get the predicted class
    pred = model.predict(img)
    pred_class = np.argmax(pred)

    # Display the predicted class to the user
    if pred_class == 0:
        messagebox.showinfo("Prediction", "Sad!")
    else:
        messagebox.showinfo("Prediction", "Cheerful!")
# Теперь мы подключим функцию paint к холсту с помощью bind,
canvas.bind("<B1-Motion>", paint)
def clear_canvas():
    canvas.delete("all")
# Мы также создадим кнопку, которая при нажатии вызывает функцию predict_smiley.
predict_button = Button(root, text="Check", command=predict_smiley)
clear_button = Button(root, text="Clear", command=clear_canvas)
clear_button.pack(side=LEFT, padx=10)
predict_button.pack()

# Run the GUI
if __name__ == "__main__":
    root.mainloop()




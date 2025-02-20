import numpy as np
import sys
sys.path.append("main.py")  # Добавь путь к папке с main.py
from main import model, SIZE

def test_prediction():
    # Создаём случайный "грустный" смайлик
    test_img = np.zeros((SIZE, SIZE, 3))
    test_img = test_img.reshape(1, SIZE, SIZE, 3) / 255.0

    # Предсказываем класс
    pred = model.predict(test_img)
    pred_class = np.argmax(pred)

    # Проверяем, что предсказание в пределах [0,1]
    assert pred_class in [0, 1], "Неверный класс!"
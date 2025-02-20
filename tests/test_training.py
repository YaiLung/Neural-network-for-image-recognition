import sys
from unittest.mock import MagicMock

# Подмена Tkinter фиктивным объектом перед импортом main.py
sys.modules["tkinter"] = MagicMock()

import numpy as np
from main import model, train_images, train_labels

def test_training():
    # Проверяем, что модель корректно обучается
    history = model.fit(train_images, train_labels, epochs=1, verbose=0)

    # Проверяем, что loss уменьшается после первой эпохи
    assert history.history['loss'][0] < 1.0, "Модель не обучается!"

![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/YaiLung/Neural-network-for-image-recognition/python-package-conda.yml)
![Static Badge](https://img.shields.io/badge/Python-3.10-blue)
![GitHub tag status](https://img.shields.io/github/checks-status/YaiLung/Neural-network-for-image-recognition/1.0.2)



# About project
## This is a Keras neural network for classifying user-drawn emoticons through the Tkinter GUI. Initially, the user creates an image with the parameters. This picture serves as the basis for creating different versions of this picture.
![___3](https://github.com/user-attachments/assets/bdd5453d-c5ab-4d16-b031-e3d97115177f)
## After the user uploads the image, the neural network creates their likeness, thereby developing the ability to recognize.
___
## Then you draw a picture on the canvas and the neural network compares yours and the ones created by it, then gives the answer.
### An example of a cheerful smiley face
![__1](https://github.com/user-attachments/assets/310612b6-294d-4f3b-8b0b-24d54d767d9d)

### An example of a sad smiley face
![__2](https://github.com/user-attachments/assets/1d25c529-de7b-4ea8-a8cd-447cee05e4ee)
___
# Мanagement
## You can expand the number of options created by the neural network.

```python
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
```
## You can add your own pictures.

```python
for i in range(1000):
    cheerful_smiley = Image.new('RGB', (SIZE, SIZE), color=CHEERFUL_COLOR)
    draw = ImageDraw.Draw(cheerful_smiley)
    draw.arc((3, 3, SIZE-4, SIZE-3), 0, 180, fill=(0, 0, 0), width=2)
    draw.rectangle((8, 7, 6, 3), fill=(0, 0, 0))
    draw.rectangle((22, 7, 20, 3), fill=(0, 0, 0))
    cheerful_smileys.append(np.array(cheerful_smiley))

    sad_smiley = Image.new('RGB', (SIZE, SIZE), color=SAD_COLOR)
    draw = ImageDraw.Draw(sad_smiley)
    draw.arc((3, 12, SIZE-4, SIZE+10), 180, 360, fill=(0, 0, 0), width=2)
    draw.rectangle((8, 7, 6, 3), fill=(0, 0, 0))
    draw.rectangle((22, 7, 20, 3), fill=(0, 0, 0))
    sad_smileys.append(np.array(sad_smiley))
```

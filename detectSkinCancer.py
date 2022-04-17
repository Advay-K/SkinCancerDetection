import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers, losses
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2

def detect(image):

    path = 'skin_cancer_data'
    train_dir = os.path.join(path, 'train')
    test_dir = os.path.join(path, 'test')
    #train
    train_benign = os.path.join(train_dir, 'benign or not skin cancer')
    train_malignant = os.path.join(train_dir, 'malignant skin cancer')
    #test
    test_benign = os.path.join(test_dir, 'benign or not skin cancer')
    test_malignant = os.path.join(test_dir, 'malignant skin cancer')

    train_img_gen = ImageDataGenerator(rescale = 1./255)
    test_img_gen = ImageDataGenerator(rescale = 1./255)

    generate_train = train_img_gen.flow_from_directory(batch_size=128, directory=train_dir, shuffle=True, target_size = (224, 224),
                                                       class_mode='binary')

    generate_test = test_img_gen.flow_from_directory(batch_size=128, directory=test_dir, target_size = (224, 224), class_mode='binary')

    #neural network

    model = Sequential([
        layers.Conv2D(16, (3,3), activation = 'relu', input_shape = (224, 224, 3)),
        layers.MaxPooling2D(pool_size = (2, 2)),
        layers.Flatten(),
        layers.Dense(10, activation= 'relu'),
        layers.Dense(1, activation = 'sigmoid')
    ])

    model.summary()

    model.compile(optimizer = 'adam', loss = losses.BinaryCrossentropy(), metrics = ['accuracy'])

    total_train_size = len(os.listdir(train_benign)) + len(os.listdir(train_malignant))
    total_test_size = len(os.listdir(test_benign)) + len(os.listdir(test_malignant))


    model.fit(generate_train, batch_size = 128, epochs = 1, steps_per_epoch = total_train_size // 128,
              validation_data = generate_test, validation_steps = total_test_size // 128)


    result = model.predict(image)
    return result
    # if result[0][0] == 1:
    #     return 'malignant'
    # else:
    #     return 'benign'
    #
    # return None






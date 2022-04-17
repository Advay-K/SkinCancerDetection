import cv2
import detectSkinCancer
from tensorflow.keras.preprocessing import image
import os
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.applications.resnet50 import preprocess_input
from tkinter import messagebox


cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()

    cv2.imshow('frame', frame)

    key = cv2.waitKey(1)

    if key == 83 or key == 115:
        cv2.imshow("frame", frame)
        cv2.imwrite("smt/test_image.png", frame)

        cv2.destroyAllWindows()
        break


img = image.load_img(os.path.join('smt', 'test_image.png'), target_size=(224, 224))

plt.imshow(img)
plt.show()


img_array = image.img_to_array(img)
img_batch = np.expand_dims(img_array, axis=0)
img_preprocessed = preprocess_input(img_batch)

result = detectSkinCancer.detect(img_preprocessed)[0][0]

if (result > 0.6):
    messagebox.showinfo('Detection Result', 'most likely benign/no skin cancer')
else:
    messagebox.showinfo('Detection Result', 'most likely malignant skin cancer')



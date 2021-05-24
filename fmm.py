import os
import pathlib
import cv2
import numpy as np
from tensorflow.keras.models import load_model


class FMM(object):
    def __init__(self, model_root=None):

        root = pathlib.Path(__file__).parent.absolute()
        model_path = os.path.join(root, 'Model/FMM.h5')

        if model_root:
            model_path = model_path

        self.model = load_model(model_path)
        self.cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        self.results = {
            0: '\tWearing Mask',
            1: '\tNo Mask'
        }

        self.colors = {
            0: (0, 255, 0),
            1: (242, 72, 75)
        }

        self.IMG_ROWS = 112
        self.IMG_COLS = 112

    def predict(self, img):
        '''
        Analyzes an image for human faces and predicts if face 
        mask is present on them
        '''
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.cascade.detectMultiScale(img_gray, 1.3, 5)

        for (x, y, w, h) in faces:
            img_face = img_gray[y:y + w, x:x + w]
            img_norm = cv2.resize(
                img_face, (self.IMG_ROWS, self.IMG_COLS)) / 255.0
            img_rshp = np.reshape(
                img_norm, (1, self.IMG_ROWS, self.IMG_COLS, 1))

            result = self.model.predict(img_rshp)
            label = np.argmax(result, axis=1)[0]

            cv2.rectangle(img, (x, y), (x + w, y + h), self.colors[label], 2)
            cv2.rectangle(img, (x, y - 40), (x + w, y), self.colors[label], -1)
            cv2.putText(img, self.results[label], (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        return img

    def monitor_camera(self, cam=0):
        '''
        Monitors the given camera for face masks, default cam is set to 0, 
        which will use the inbuilt web cam
        '''
        video = cv2.VideoCapture(cam)
        while True:
            ret, img = video.read()
            img = self.predict(img)
            cv2.imshow('Face Mask Monitor - Live', img)

            k = cv2.waitKey(1)
            if k == 27:
                break

    def check_image(self, img):
        img = cv2.imread(img, cv2.IMREAD_UNCHANGED)
        return self.predict(img)

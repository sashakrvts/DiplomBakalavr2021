import cv2
import numpy as np
from keras.datasets import mnist
from keras.layers import Dense, Flatten
from keras.layers.convolutional import Conv2D
from keras.models import Sequential
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import pickle
image = cv2.imread('Images/MultiDigitsTesting/testing3.jpg')
def image_processing(image):
        loaded_model = open("trained_model.p", "rb")
        model = pickle.load(loaded_model)
        text_num = []
        #image = cv2.imread('./test5.jpg')
        gray = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
        ret, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        contours = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

        for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                #print(w)
                if(w>=10 and h>=10 and w<=30):
                        #print(w,h)
                        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 1)
                        digit = th[y:y + h, x:x + w]
                        resized_digit = cv2.resize(digit, (18, 18))
                        padded_digit = np.pad(resized_digit, ((5, 5), (5, 5)), "constant", constant_values=0)
                        #print(padded_digit.shape)
                        digit = padded_digit.reshape(1, 28, 28, 1)
                        digit = digit / 255.0

                        pred = model.predict([digit])[0]
                        final_pred = np.argmax(pred)
                        text_num.append([x, final_pred])


                        data = str(final_pred) + ' ' + str(int(max(pred) * 100)) + '%'

                        font = cv2.FONT_HERSHEY_SIMPLEX
                        fontScale = 0.5
                        color = (0, 0, 0)
                        thickness = 1
                        cv2.putText(image, data, (x, y - 5), font, fontScale, color, thickness)

        text_num = sorted(text_num, key=lambda t: t[0])
        text_num = [i[1] for i in text_num]
        final_text = "".join(map(str, text_num))
        return final_text

answer = image_processing(image)
#cv2.imshow(answer , image)
#print(answer)
#print(final_text)
#cv2.imshow('image', image)
cv2.waitKey(0)
import tensorflow as tf
import streamlit as st
import pickle
import cv2
from PIL import Image, ImageOps
import numpy as np
import Forms_Main
import pandas as pd
import MultiDigits
from threading import Thread

import keras.backend.tensorflow_backend as tb
tb._SYMBOLIC_SCOPE.value = True

st.set_option('deprecation.showfileUploaderEncoding', False)
@st.cache(allow_output_mutation = True)
def load_model():
    loaded_model = open("trained_model.p", "rb")
    model = pickle.load(loaded_model)
    return model
# t = Thread(target=load_model)
# t.start()
# t.join()

model = load_model()
st.write('''
         # Перевірка відповідей
         ''')
file_right_answers =  st.file_uploader('Завантажте файл з правильними відповідями',
                         type = ['xlsx', 'txt'])
file_blank =  st.file_uploader('Завантажте фотографію бланка з відповідями',
                         type = ['jpg', 'png'])

def import_and_predict(image):
    #size = [1280, 590]
    #image = ImageOps.fit(image, Image.ANTIALIAS)
    print("import predict 1", type(image))
    image = np.asarray(image)
    print("import predict 2", type(image)) #str
    #img_reshape = image.reshape((image.shape[0], 1280, 590, 3)).astype('float32')
    answers = Forms_Main.blank_crop(image)
    return answers



if file_blank is None or file_right_answers is None:
    st.text("Завантажте обидва файли для проведення перевірки")
else:

    image_blank = Image.open(file_blank)
    file_answers =  pd.read_excel(file_right_answers)
    file_answers=file_answers.to_numpy()
    print(file_answers)
    st.success(file_answers)
    print("main 1", type(image_blank))
    image_blank = np.asarray(image_blank)
    print("main 2", type(image_blank))  # str
    st.image(image_blank) # Image display
    #ans = import_and_predict(image)
    answers = Forms_Main.blank_crop(image_blank)
    st.success(answers)
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
                         type = ['xlsx'])
file_blank =  st.file_uploader('Завантажте фотографію бланка з відповідями',
                         type = ['jpg', 'png'])


def receive_answers_from_excel(file_right_answers):
    right_answers = pd.read_excel(file_right_answers)
    right_answers = right_answers.to_numpy()
    only_right_answers = []
    for el in right_answers:
        only_right_answers.append(el[1])
    return only_right_answers

def receive_blank_answers(file_blank):
    image_blank = Image.open(file_blank)
    image_blank = np.asarray(image_blank)
    st.image(image_blank) # Image display
    blank_answers = Forms_Main.blank_crop(image_blank)
    return blank_answers



if file_blank is None or file_right_answers is None:
    st.text("Завантажте обидва файли для проведення перевірки")
else:

    right_answers = receive_answers_from_excel((file_right_answers))
    st.success("Правильні відповіді: "+ str(right_answers))

    blank_answers = receive_blank_answers(file_blank)
    st.success("Відповіді з бланку: " + str(blank_answers))
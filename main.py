import tensorflow as tf
import streamlit as st
import pickle
import cv2
from PIL import Image, ImageOps
import numpy as np
import Forms_Main
import pandas as pd
import marks_estimation
import MultiDigits
from threading import Thread

import keras.backend.tensorflow_backend as tb

tb._SYMBOLIC_SCOPE.value = True

st.set_option('deprecation.showfileUploaderEncoding', False)


@st.cache(allow_output_mutation=True)
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
file_right_answers = st.file_uploader('Завантажте файл з правильними відповідями',
                                      type=['xlsx'])
file_blank = st.file_uploader('Завантажте фотографію бланка з відповідями',
                              type=['jpg', 'png'])


def receive_answers_from_excel(file_right_answers):
    right_answers = pd.read_excel(file_right_answers)
    right_answers = right_answers.to_numpy()
    only_right_answers = []
    for el in right_answers:
        only_right_answers.append(el[1])
    return only_right_answers


def receive_blank_answers(file_blank,num):
    image_blank = Image.open(file_blank)
    image_blank = np.asarray(image_blank)
    st.image(image_blank)  # Image display
    blank_answers = Forms_Main.blank_crop(image_blank,num)
    return blank_answers

def error_check(blank_answers):
    count1,count2, count3 = 0, 0, 0
    user_answer = st.radio("Чи всі відповіді вірно розпізнані?",
                           ('Так', 'Ні'), key=count1)
    while user_answer == 'Ні':
        st.write('Вкажіть номер відповіді, яку розпізнано невірно')
        user_answer2 = int(st.number_input(label='Номер відповіді', step=1.0,key = count2))
        print(user_answer2, type(user_answer2))
        user_answer3 = st.text_input(label='Значення відповіді з бланку', key = count3)
        if user_answer2 is not None and user_answer3 is not None:
            blank_answers[user_answer2 - 1] = user_answer3
        count1 += 1
        count1 += 2
        count1 += 3
        user_answer = st.radio("Чи всі останні відповіді вірно розпізнані?",
                               ('Так', 'Ні'), key=count1)
        # user_answer = st.radio("Чи всі відповіді вірно розпізнані?",
        # ('Так', 'Ні'))
        return  blank_answers


if file_blank is None or file_right_answers is None:
    st.text("Завантажте обидва файли для проведення перевірки")
else:

    right_answers = receive_answers_from_excel((file_right_answers))
    st.text("Правильні відповіді: " + str(right_answers))
    num_right_answers = len(right_answers)

    blank_answers = receive_blank_answers(file_blank, num_right_answers)
    st.text("Відповіді з бланку: " + str(blank_answers))

    # count1, count2, count3 = 0, 1, 2
    # user_answer = st.radio("Чи всі відповіді вірно розпізнані?",
    #                        ('Так', 'Ні'), key=count1)
    # while user_answer == 'Ні':
    #     st.write('Вкажіть номер відповіді, яку розпізнано невірно')
    #     user_answer2 = int(st.number_input(label='Номер відповіді', step=1.0, key=count2))
    #     print(user_answer2, type(user_answer2))
    #     user_answer3 = st.text_input(label='Значення відповіді з бланку', key=count3)
    #     if user_answer2 is not None and user_answer3 is not None:
    #         blank_answers[user_answer2 - 1] = user_answer3
    #     count1 += 1
    #     count2 += 1
    #     count3 += 1
    #     user_answer = st.radio("Чи всі останні відповіді вірно розпізнані?",
    #                            ('Так', 'Ні'), key=count1)

    # blank_answers=error_check(blank_answers)
    # st.success("Відповіді з бланку: " + str(blank_answers))


    st.success('Відсоток правильних відповідей студента ' +
               str(blank_answers[0] + ' складає '
                   + str(marks_estimation.calculate_mark(blank_answers, right_answers))))
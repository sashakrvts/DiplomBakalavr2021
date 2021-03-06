from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
import pickle
import matplotlib.pyplot as plt
import numpy as np

(X_train, y_train), (X_test, y_test) = mnist.load_data()

print("Сумарна кількість зображень з набору даних MNIST: ",(X_train.shape[0]+X_test.shape[0]))
print("Кількість класів зображень з набору даних MNIST: ",10)
print("Структура даних для тренувального набору:")
print("Структура масиву з матрицями зображень: ",X_train.shape)
print("Структура масиву з ярликами(labels): ", y_train.shape)
print("Структура даних для тестувального набору:")
print("Структура масиву з матрицями зображень: ",X_test.shape)
print("Структура масиву з ярликами(labels): ",y_test.shape)

X_train = X_train.reshape((X_train.shape[0], 28, 28, 1)).astype('float32')
X_test = X_test.reshape((X_test.shape[0], 28, 28, 1)).astype('float32')
samples_num = []

X_train = X_train / 255
X_test = X_test / 255

print(X_train.shape)
for x in range(10):
    #print(len(np.where(y_train==x)[0]))
    samples_num.append(len(np.where(y_train==x)[0]))



y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

plt.figure(figsize = (10,5))

plt.bar(range(num_classes),samples_num)
plt.title('Розподіл навчальної вибірки по класам')
plt.xlabel('Клас з зображеннями цифри')
plt.ylabel('Кількість зображень')
plt.show()



# def myModel():
#     model = Sequential()
#     model.add(Conv2D(30, (5, 5), input_shape=(28, 28, 1), activation="relu"))
#     model.add(MaxPooling2D())
#     model.add(Conv2D(15, (3, 3), activation="relu"))
#     model.add(MaxPooling2D())
#     model.add(Dropout(0.2))
#     model.add(Flatten())
#     model.add(Dense(128, activation="relu"))
#     model.add(Dense(50, activation="relu"))
#     model.add(Dense(num_classes, activation="softmax"))
#
#     #compilation
#     model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
#     return model
#
# model = myModel()
# history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200)
# scores = model.evaluate(X_test, y_test, verbose=0)
# print("Large CNN Error: %.2f%%" % (100-scores[1]*100))
#
# # #### PLOT THE RESULTS
# # plt.figure(1)
# # plt.plot(history.history['loss'])
# # plt.plot(history.history['val_loss'])
# # plt.legend(['Тренувальний набір', 'Тестовий набір'])
# # plt.title('Втрати')
# # plt.xlabel('Епохи (прогони тренуваняня моделі)')
# # plt.figure(2)
# # plt.plot(history.history['accuracy'])
# # plt.plot(history.history['val_accuracy'])
# # plt.legend(['Тренувальний набір', 'Тестовий набір'])
# # plt.title('Точність')
# # plt.xlabel('Епохи (прогони тренуваняня моделі)')
# # plt.show()
#
# pickle_out = open("trained_model.p", "wb")
# pickle.dump(model, pickle_out)
# pickle_out.close()
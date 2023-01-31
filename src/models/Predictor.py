from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout
from keras.optimizers import SGD, Adam

import numpy as np


class Predictor():
    def __init__(self, mnist=False, emnist=False, quickdraw=False):

        if mnist:
            self.mnist_model = Model()
            self.mnist_model.model.load_weights("src/models/mnist_model/model_hand.h5")

        if emnist:
            self.emnist_model = Model()
            self.emnist_model.model.load_weights("src/models/emnist_model/model_hand.h5")

        if quickdraw:
            self.quickdraw_model = Model()
            self.quickdraw_model.model.load_weights("src/models/quickdraw_model/model_hand.h5")
    

    def mnist(self, img):
        return self.mnist_model.predict(img)
        
    def emnist(self, img):
        return self.emnist_model.predict(img)

    def quickdraw(self, img):
        return self.quickdraw_model.predict(img)



class Model():
    def __init__(self):
        self.build_network()

    def build_network(self):
        self.model = Sequential()

        self.model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(28,28,1)))
        self.model.add(MaxPool2D(pool_size=(2, 2), strides=2))

        self.model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding = 'same'))
        self.model.add(MaxPool2D(pool_size=(2, 2), strides=2))

        self.model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding = 'valid'))
        self.model.add(MaxPool2D(pool_size=(2, 2), strides=2))

        self.model.add(Flatten())

        self.model.add(Dense(128,activation ="relu"))
        self.model.add(Dense(256,activation ="relu"))

        self.model.add(Dense(26,activation ="softmax"))

        self.model.compile(optimizer = Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])


    def predict(self, img):
        return np.argmax(self.model(np.array([img])))


from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout
from keras.optimizers import SGD, Adam
import numpy as np

class Predictor():
    def __init__(self, mnist=False, emnist=False, quickdraw=False):

        if mnist:
            self.mnist_model = Model(10)
            self.mnist_model.model.load_weights("src/models/mnist_model/model_hand.h5")

        if emnist:
            self.emnist_model = Model(26)
            self.emnist_model.model.load_weights("src/models/emnist_model/model_hand.h5")

        if quickdraw:
            self.quickdraw_model = Model(26)
            self.quickdraw_model.model.load_weights("src/models/quickdraw_model/model_hand.h5")
    
    # Mnist Predictions
    def mnist(self, img, mode="hard"):
        if mode == "hard":
            return self.mnist_model.hard_predict(img)
        elif mode == "test":
            return self.mnist_model.test_predict(img)
        else:
            return self.mnist_model.soft_predict(img)

    # Emnist Predictions
    def emnist(self, img, mode="hard"):
        if mode == "hard":
            return self.emnist_model.hard_predict(img)
        elif mode == "test":
            return self.emnist_model.test_predict(img)
        else:
            return self.emnist_model.soft_predict(img)

    # Quickdraw Predictions
    def quickdraw(self, img, mode="hard"):
        if mode == "hard":
            return self.quickdraw_model.hard_predict(img)
        elif mode == "test":
            return self.quickdraw_model.test_predict(img)
        else:
            return self.quickdraw_model.soft_predict(img)


class Model():
    def __init__(self, classes : int):
        self.build_network(classes)

    def build_network(self, classes : int):
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

        self.model.add(Dense(classes,activation ="softmax"))

        self.model.compile(optimizer = Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])


    def hard_predict(self, img):
        softmax = self.softmax_sample(self.model(np.array([img])))
        if np.max(softmax) < 0.75: #too unsure
            return -1
        else:
            return np.argmax(softmax)
        
    def test_predict(self, img):
        softmax = self.softmax_sample(self.model(np.array([img]))) #No certainty check
        return np.argmax(softmax)
        
    def soft_predict(self, img):
        return self.softmax_sample(self.model(np.array([img])))[0] #full prediction array
       
    
    def softmax_sample(self, a, temperature=3.25):
        # helper function to sample an index from a probability array
        a = np.log(a) / temperature
        a = np.exp(a) / np.sum(np.exp(a))   
        return a



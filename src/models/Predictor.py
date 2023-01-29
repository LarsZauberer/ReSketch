import numpy as np

from models.mnist_model.models import EfficientCapsNet
import onnx
from onnx_tf.backend import prepare
from keras.models import model_from_json



class Predictor():
    def __init__(self):
        #for mnist test
        self.mnist_model = EfficientCapsNet('MNIST', mode='test', verbose=False)
        self.mnist_model.load_graph_weights()
        
        # For quickdraw test
        # reference: https://analyticsindiamag.com/converting-a-model-from-pytorch-to-tensorflow-guide-to-onnx/
        q_model = onnx.load("src/models/quickdraw_model/quickdraw.onnx")
        self.tf_q_model = prepare(q_model)
        
        # For emnist test
        json_file = open('src/models/emnist_model/model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.emnist_model = model_from_json(loaded_model_json)
        self.emnist_model.load_weights('src/models/emnist_model/model.h5')


    def mnist(self, img):
        img = np.array([img.reshape(28, 28, 1)])
        predict = self.mnist_model.predict(img)

        choice = np.argmax(predict[0][0])

        

        
        # Too unsure. Should not be validated
        if predict[0][0][choice] < 0.75:
            choice = -1
        
        return choice


    def quickdraw(self, img):
        img = img.reshape(28 * 28)
        predict = self.tf_q_model.run(np.array([img], dtype=np.float32))
        return np.argmax(predict[0])


    def emnist(self, img):
        img = img.reshape(28 * 28)
        predict = self.emnist_model(np.array([img], dtype=np.float32))
        return np.argmax(predict[0])
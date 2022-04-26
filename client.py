import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
import tensorflow as tf
import numpy as np

from tensorflow.keras.layers import Conv2D
from tensorflow.keras.models import Sequential, Model

class FLClient:
    def __init__(self, id):
        self.id = id
#        self.net = self.build_net()
#        self.data = self.pred_data("mnist")

    def prep_data(self, name):
        if name == "mnist":
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
            x_train = x_train.reshape(-1, 28, 28, 1) # infer length (-1), h, w, c
            x_test  = x_test.reshape(-1, 28, 28, 1)
            return (x_train, y_train), (x_test, y_test)
            
        if name == "cifar10":
            return tf.keras.datasets.cifar10.load_data()

        if name == "cifar100":
            return tf.keras.datasets.cifar100.load_data()
        
        if name == "fmnist":
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
            x_train = x_train.reshape(-1, 28, 28, 1)
            x_test  = x_test.reshape(-1, 28, 28, 1)
            return (x_train, y_train), (x_test, y_test)

    
    def prep_client(self, data_name):
        self.net = self.build_net()
        self.data = self.pred_data(data_name)
        
    def build_net(self):
        net = Sequential([

        ])
        return net

    def train(self, data_idxs, global_weight, epochs, batch_size):
        if global_weight != None:
            self.net.set_weights(global_weight)
        
        split_x_train, split_y_train = self.data[0][data_idxs], self.data[1][data_idxs]

        self.net.fit(split_x_train, split_y_train, epochs=epochs, batch_size=batch_size, verbose=0)
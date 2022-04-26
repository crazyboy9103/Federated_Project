import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
import tensorflow as tf
import numpy as np

from tensorflow.keras.layers import Conv2D
from tensorflow.keras.models import Sequential, Model

import logging
import threading
import random

class FLServer:
    EXP_IID = 1
    EXP_NONIID = 2
    def __init__(self):
        self.curr_round = 0
    
    def build_logger(self, name):
        logger = logging.getLogger('log_custom')
        logger.setLevel(logging.INFO)

        formatter = logging.Formatter("%(asctime)s;[%(levelname)s];%(message)s",
                              "%Y-%m-%d %H:%M:%S")
        
        streamHandler = logging.StreamHandler()
        streamHandler.setFormatter(formatter)
        streamHandler.setLevel(logging.INFO)
        logger.addHandler(streamHandler)

        fileHandler = logging.FileHandler(f'log_{name}.txt', mode = "w")
        fileHandler.setFormatter(formatter)
        fileHandler.setLevel(logging.INFO)
        logger.addHandler(fileHandler)
        
        logger.propagate = False
        return logger

    def init_server(self, data_name, experiment, num_clients, max_rounds, epochs, batch_size):
        print("Preping dataset")
        (self.x_train, self.y_train), (self.x_test, self.y_test) = self.prep_data(data_name)
    
    def prep_data(self, data_name):
        if name == "mnist":
            (_, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
            x_test = x_test.reshape(-1, 28, 28, 1)
            return (_, y_train), (x_test, y_test) 
            
        if name == "cifar10":
            return tf.keras.datasets.cifar10.load_data()

        if name == "cifar100":
            return tf.keras.datasets.cifar100.load_data()

        if name == "fmnist":
            return tf.keras.datasets.fashion_mnist.load_data()

    def split_data(self, ratio, num_server_data, num_clients):
        # fedrgd split
        # https://arxiv.org/pdf/2008.11364.pdf @ appendix A.2
        

        


        if experiment == self.EXP_IID:
            
        
        
        if experiment == self.EXP_NONIID:

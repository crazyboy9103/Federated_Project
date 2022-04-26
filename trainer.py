from client import *
from server import * 
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
import tensorflow as tf
import numpy as np

from tensorflow.keras.layers import Conv2D
from tensorflow.keras.models import Sequential, Model

class MyClient(FLClient):
    
    def build_net(self):
        # creates self.net
        return net
    
    def train(self, data_idxs, global_weight, epochs, batch_size):
        # updates self.net
    
    def evaluate(self):
        # evaluate self.net


class MyServer(FLServer):

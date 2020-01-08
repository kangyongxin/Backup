# -*- coding: utf-8 -*


#对agent,soul 进行顶层封装

import tensorflow as tf
import numpy as np

from SOUL3 import SOUL3
from _AGENT3 import AGENT3

class MINE:
    def __init__(self,env, config):
        self.config = config
        self.sess = tf.InteractiveSession()
        self.agent = AGENT3(env, self.config)
        self.soul = SOUL3(env, self.config)
        self.saver = tf.train.Saver()




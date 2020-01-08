# -*- coding: utf-8 -*
import tensorflow as tf
import numpy as np
import random
import functools
from my_Memory import Memory
from policy_net import Policy_net
from ppo import PPOTrain
from discriminator import Discriminator


def lazy_property(func):
    attribute = '_lazy_' + func.__name__

    @property
    @functools.wraps(func)
    def wrapper(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, func(self))
        return getattr(self, attribute)
    return wrapper

class SOUL3:
    def __init__(self, env, config, sess):
        # self.Soulsess = tf.InteractiveSession()
        if sess is None:
            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())
        else:
            self.sess = sess
        self.config = config
        self.expert_memory = Memory(capacity=self.config.EXPERT_MEMORY_SIZE,permanent_data=0)
        self.generate_memory = Memory(capacity=self.config.GENERATE_MEMORY_SIZE, permanent_data=0)
        #self.sess.run(tf.global_variables_initializer())

        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        self.ob_space = env.observation_space
        self.gamma = 0.95
        self.Policy = Policy_net('policy', env)
        self.Old_Policy = Policy_net('old_policy', env)
        self.PPO = PPOTrain(self.Policy, self.Old_Policy, self.gamma)
        self.D = Discriminator(env)
        self.epsilon = self.config.INITIAL_EPSILON
        self.saver = tf.train.Saver()

    def add_data_to_memory(self, D, M):
        for t in D:
            M.store(np.array(t, dtype=object))

    def get_data_from_fullmemory(self, MFULL):
        _, D, _ = MFULL.sample(len(MFULL))
        return D

    def copy_AFULL_to_B(self, AFULL, B):
        _, data, _ = AFULL.sample(len(AFULL))
        for t in data:
            B.store(np.array(t, dtype=object))

    def random_action(self):
        return random.randint(0, self.action_dim - 1)

    def perceive(self, transition):
        self.generate_memory.store(np.array(transition))
        # epsilon->FINAL_EPSILON(min_epsilon)
        if self.generate_memory.full():
            self.epsilon = max(self.config.FINAL_EPSILON, self.epsilon * self.config.EPSILIN_DECAY)

    def save_model(self):
        print("soul Model saved in : {}".format(self.saver.save(self.sess, self.config.SOUL_MODEL_PATH)))

    def restore_model(self):
        self.saver.restore(self.sess, self.config.SOUL_MODEL_PATH)
        print("soul Model restored.")

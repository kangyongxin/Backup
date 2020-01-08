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

class GAIL:
    def __init__(self, config, env, demo_transitions=None): #we need another file to give the defination of configuration
        self.sess = tf.InteractiveSession()
        self.config = config
        self.generate_memory = Memory(capacity=self.config.generate_memory_size, permanent_data=0)
        self.expert_memory = Memory(capacity=self.config.expert_memory_size, permanent_data=0)
        self.add_data_to_genarte_memory(source=demo_transitions)
        self.add_data_to_expert_memory(source=demo_transitions)
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        self.ob_space = env.observation_space
        self.gamma=0.95
        self.Policy = Policy_net('policy', env)
        self.Old_Policy = Policy_net('old_policy', env)
        self.PPO = PPOTrain( self.Policy,self.Policy,self.gamma)
        self.D = Discriminator(env)

        self.saver = tf.train.Saver()

        self.sess.run(tf.global_variables_initializer())
        print("we have initialized the GAIL")

        self.save_model()
        self.restore_model()

    def add_data_to_genarte_memory(self, source):
        for t in source:
            self.generate_memory.store(np.array(t, dtype=object))

    def add_data_to_expert_memory(self, source):
        for t in source:
            self.expert_memory.store(np.array(t, dtype=object))

    def perceive(self, transition):
        self.generate_memory.store(np.array(transition))
        # epsilon->FINAL_EPSILON(min_epsilon)
        if self.generate_memory.full():
            self.epsilon = max(self.config.FINAL_EPSILON, self.epsilon * self.config.EPSILIN_DECAY)

    def save_model(self):
        print("Model saved in : {}".format(self.saver.save(self.sess, self.config.MODEL_PATH_soul)))

    def restore_model(self):
        self.saver.restore(self.sess, self.config.MODEL_PATH_soul)
        print("Model restored.")

# -*- coding: utf-8 -*
import tensorflow as tf
import numpy as np
from my_Memory3 import Memory
import  functools
import random


def lazy_property(func):
    attribute = '_lazy_' + func.__name__

    @property
    @functools.wraps(func)
    def wrapper(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, func(self))
        return getattr(self, attribute)
    return wrapper




class AGENT3:
    def __init__(self, env, config):
        self.config = config
        self.Agentsess = tf.InteractiveSession()
        self.demo_memory = Memory(capacity=self.config.DEMO_BUFFER_SIZE, permanent_data= self.config.DEMO_BUFFER_SIZE)
        self.replay_memory = Memory(capacity= self.config.REPLAY_BUFFER_SIZE, permanent_data= self.config.DEMO_BUFFER_SIZE)

        self.epsilon = self.config.INITIAL_EPSILON

        self.action_dim = env.action_space.n
        self.state_dim = env.observation_space.shape[0]
        self.action_batch = tf.placeholder("int32", [None])
        self.y_input = tf.placeholder("float", [None, self.action_dim])
        self.ISWeights = tf.placeholder("float", [None, 1])
        self.n_step_y_input = tf.placeholder("float", [None, self.action_dim])  # for n-step reward
        self.isdemo = tf.placeholder("float", [None])
        self.eval_input = tf.placeholder("float", [None, self.state_dim])
        self.select_input = tf.placeholder("float", [None, self.state_dim])


        self.Q_eval
        self.Q_select

        self.loss
        self.optimize
        self.update_target_net
        self.abs_errors

        self.saver = tf.train.Saver()

        self.Agentsess.run(tf.global_variables_initializer())
        self.time_step = 0

        self.save_model()
        self.restore_model()


    def add_data_to_memory(self, D , M):
        for t in D:
            M.store(np.array(t,dtype=object))


    def get_data_from_fullmemory(self,  MFULL):
        _, D, _ = MFULL.sample(len(MFULL))
        return D

    def copy_AFULL_to_B(self, AFULL, B):
        _, data, _ = AFULL.sample(len(AFULL))
        for t in data:
            B.store(np.array(t, dtype=object))

    def random_action(self):
        return random.randint(0,self.action_dim- 1)

    def pre_train(self):
        print('Pre-training ...')
        for i in range(self.config.PRETRAIN_STEPS):
            self.train_Q_network(pre_train=True)
            if i % 200 == 0 and i > 0:
                print('{} th step of pre-train finish ...'.format(i))
        self.time_step = 0
        print('All pre-train finish.')

    def build_layers(self, state, c_names, units_1, units_2, w_i, b_i, reg=None):
        a_d = self.action_dim
        with tf.variable_scope('l1'):
            #w1 = tf.get_variable('w1', [a_d, units_1], initializer=w_i, collections=c_names, regularizer=reg)#self.state_dim
            w1 = tf.get_variable('w1', [self.state_dim, units_1], initializer=w_i, collections=c_names, regularizer=reg)#20190122
            b1 = tf.get_variable('b1', [1, units_1], initializer=b_i, collections=c_names, regularizer=reg)
            dense1 = tf.nn.relu(tf.matmul(state, w1) + b1)
        with tf.variable_scope('l2'):
            w2 = tf.get_variable('w2', [units_1, units_2], initializer=w_i, collections=c_names, regularizer=reg)
            b2 = tf.get_variable('b2', [1, units_2], initializer=b_i, collections=c_names, regularizer=reg)
            dense2 = tf.nn.relu(tf.matmul(dense1, w2) + b2)
        with tf.variable_scope('l3'):
            w3 = tf.get_variable('w3', [units_2, a_d], initializer=w_i, collections=c_names, regularizer=reg)
            b3 = tf.get_variable('b3', [1, a_d], initializer=b_i, collections=c_names, regularizer=reg)
            dense3 = tf.matmul(dense2, w3) + b3
        return dense3

    @lazy_property
    def Q_select(self):
        with tf.variable_scope('select_net') as scope:
            c_names = ['select_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
            w_i = tf.random_uniform_initializer(-0.1, 0.1)
            b_i = tf.constant_initializer(0.1)
            reg = tf.contrib.layers.l2_regularizer(scale=0.2)  # Note: only parameters in select-net need L2
            return self.build_layers(self.select_input, c_names, 24, 24, w_i, b_i, reg)

    @lazy_property
    def Q_eval(self):
        with tf.variable_scope('eval_net') as scope:
            c_names = ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
            w_i = tf.random_uniform_initializer(-0.1, 0.1)
            b_i = tf.constant_initializer(0.1)
            return self.build_layers(self.eval_input, c_names, 24, 24, w_i, b_i)

    def loss_l(self, ae, a):
        return 0.0 if ae == a else 0.8

    def loss_jeq(self, Q_select):
        jeq = 0.0
        for i in range(self.config.BATCH_SIZE):
            ae = self.action_batch[i]
            max_value = float("-inf")
            for a in range(self.action_dim):
                max_value = tf.maximum(Q_select[i][a] + self.loss_l(ae, a), max_value)
            jeq += self.isdemo[i] * (max_value - Q_select[i][ae])
        return jeq

    @lazy_property
    def loss(self):
        l_dq = tf.reduce_mean(tf.squared_difference(self.Q_select, self.y_input))
        l_n_dq = tf.reduce_mean(tf.squared_difference(self.Q_select, self.n_step_y_input))
        # l_n_step_dq = self.loss_n_step_dq(self.Q_select, self.n_step_y_input)
        l_jeq = self.loss_jeq(self.Q_select)
        l_l2 = tf.reduce_sum([tf.reduce_mean(reg_l) for reg_l in tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)])
        return self.ISWeights * tf.reduce_sum([l * λ for l, λ in zip([l_dq, l_n_dq, l_jeq, l_l2], self.config.LAMBDA)])

    @lazy_property
    def abs_errors(self):
        return tf.reduce_sum(tf.abs(self.y_input - self.Q_select), axis=1)  # only use 1-step R to compute abs_errors

    @lazy_property
    def optimize(self):
        optimizer = tf.train.AdamOptimizer(self.config.LEARNING_RATE)
        return optimizer.minimize(self.loss)  # only parameters in select-net is optimized here

    @lazy_property
    def update_target_net(self):
        select_params = tf.get_collection('select_net_params')
        eval_params = tf.get_collection('eval_net_params')
        return [tf.assign(e, s) for e, s in zip(eval_params, select_params)]

    def save_model(self):
        print("Model saved in : {}".format(self.saver.save(self.Agentsess, self.config.AGENT_MODEL_PATH)))

    def restore_model(self):
        self.saver.restore(self.Agentsess, self.config.AGENT_MODEL_PATH)
        print("Model restored.")

    def perceive(self, transition):
        self.replay_memory.store(np.array(transition))
        # epsilon->FINAL_EPSILON(min_epsilon)
        if self.replay_memory.full():
            self.epsilon = max(self.config.FINAL_EPSILON, self.epsilon * self.config.EPSILIN_DECAY)

    def train_Q_network(self, pre_train=False, update=True):
        """
        :param pre_train: True means should sample from demo_buffer instead of replay_buffer
        :param update: True means the action "update_target_net" executes outside, and can be ignored in the function
        """
        if not pre_train and not self.replay_memory.full():  # sampling should be executed AFTER replay_memory filled
            return
        self.time_step += 1

        assert self.replay_memory.full() or pre_train

        actual_memory = self.demo_memory if pre_train else self.replay_memory
        tree_idxes, minibatch, ISWeights = actual_memory.sample(self.config.BATCH_SIZE)
        # todo 在此引入其他采样方法/其他的采样方法是要依赖于 其他构造树的方法。现在的想法是先把这个接口暴露出来

        #actual_memory.memory_test()

        #

        np.random.shuffle(minibatch)
        state_batch = [data[0] for data in minibatch]
        action_batch = [data[1] for data in minibatch]
        reward_batch = [data[2] for data in minibatch]
        next_state_batch = [data[3] for data in minibatch]
        done_batch = [data[4] for data in minibatch]
        demo_data = [data[5] for data in minibatch]
        n_step_reward_batch = [data[6] for data in minibatch]
        n_step_state_batch = [data[7] for data in minibatch]
        n_step_done_batch = [data[8] for data in minibatch]
        actual_n = [data[9] for data in minibatch]

        # provide for placeholder，compute first
        Q_select = self.Q_select.eval(feed_dict={self.select_input: next_state_batch})
        Q_eval = self.Q_eval.eval(feed_dict={self.eval_input: next_state_batch})
        n_step_Q_select = self.Q_select.eval(feed_dict={self.select_input: n_step_state_batch})
        n_step_Q_eval = self.Q_eval.eval(feed_dict={self.eval_input: n_step_state_batch})

        y_batch = np.zeros((self.config.BATCH_SIZE, self.action_dim))
        n_step_y_batch = np.zeros((self.config.BATCH_SIZE, self.action_dim))
        for i in range(self.config.BATCH_SIZE):
            # state, action, reward, next_state, done, demo_data, n_step_reward, n_step_state, n_step_done = t
            temp = self.Q_select.eval(feed_dict={self.select_input: state_batch[i].reshape((-1, self.state_dim))})[0]
            temp_0 = np.copy(temp)
            # add 1-step reward
            action = np.argmax(Q_select[i])
            temp[action_batch[i]] = reward_batch[i] + (1 - int(done_batch[i])) * self.config.GAMMA * Q_eval[i][action]
            y_batch[i] = temp
            # add n-step reward
            action = np.argmax(n_step_Q_select[i])
            q_n_step = (1 - int(n_step_done_batch[i])) * self.config.GAMMA**actual_n[i] * n_step_Q_eval[i][action]
            temp_0[action_batch[i]] = n_step_reward_batch[i] + q_n_step
            n_step_y_batch[i] = temp_0
        #tf.summary.FileWriter("logs/", self.sess.graph)  #
        #print("train Q network")
        #print("self.y_input: ",y_batch, "self.n_step_y_input:", n_step_y_batch, "self.select_input:", state_batch, "self.action_batch:", action_batch,"self.isdemo:",demo_data,"self.ISWeights: ", ISWeights)
        _, abs_errors = self.Agentsess.run([self.optimize, self.abs_errors],
                                      feed_dict={self.y_input: y_batch,
                                                 self.n_step_y_input: n_step_y_batch,
                                                 self.select_input: state_batch,
                                                 self.action_batch: action_batch,
                                                 self.isdemo: demo_data,
                                                 self.ISWeights: ISWeights})

        self.replay_memory.batch_update(tree_idxes, abs_errors)  # update priorities for data in memory

        # 此例中一局步数有限，因此可以外部控制一局结束后update ，update为false时在外部控制
        if update and self.time_step % self.config.UPDATE_TARGET_NET == 0:
            self.Agentsess.run(self.update_target_net)


    def egreedy_action(self, state):
        if random.random() <= self.epsilon:
            return random.randint(0, self.action_dim - 1)
        return np.argmax(self.Q_select.eval(feed_dict={self.select_input: [state]})[0])

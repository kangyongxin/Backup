"""
This part of code is the reinforcement learning brain, which is a brain of the agent.
All decisions are made in here.

Policy Gradient, Reinforcement Learning.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
Tensorflow: 1.0
gym: 0.8.0
"""

import numpy as np
import tensorflow as tf
import pandas as pd

# reproducible
# np.random.seed(1)
# tf.set_random_seed(1)


class PolicyGradient:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.01,
            reward_decay=0.95,
            output_graph=False,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay

        self.ep_obs, self.ep_as, self.ep_rs = [], [], []

        self._build_net()

        self.sess = tf.Session()

        if output_graph:
            # $ tensorboard --logdir=logs
            # http://0.0.0.0:6006/
            # tf.train.SummaryWriter soon be deprecated, use following
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())

    def _build_net(self):
        with tf.name_scope('inputs'):
            self.tf_obs = tf.placeholder(tf.float32, [None, self.n_features], name="observations")
            self.tf_acts = tf.placeholder(tf.int32, [None, ], name="actions_num")
            self.tf_vt = tf.placeholder(tf.float32, [None, ], name="actions_value")
        # fc1
        layer = tf.layers.dense(
            inputs=self.tf_obs,
            units=10,
            activation=tf.nn.tanh,  # tanh activation
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            name='fc1'
        )
        # fc2
        all_act = tf.layers.dense(
            inputs=layer,
            units=self.n_actions,
            activation=None,
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            name='fc2'
        )

        self.all_act_prob = tf.nn.softmax(all_act, name='act_prob')  # use softmax to convert to probability

        with tf.name_scope('loss'):
            # to maximize total reward (log_p * R) is to minimize -(log_p * R), and the tf only have minimize(loss)
            neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=all_act, labels=self.tf_acts)   # this is negative log of chosen action
            # or in this way:
            # neg_log_prob = tf.reduce_sum(-tf.log(self.all_act_prob)*tf.one_hot(self.tf_acts, self.n_actions), axis=1)
            loss = tf.reduce_mean(neg_log_prob * self.tf_vt)  # reward guided loss

        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)

    def choose_action(self, observation):
        #print("observation",observation)
        #print("np.newaxis",np.newaxis)
        prob_weights = self.sess.run(self.all_act_prob, feed_dict={self.tf_obs: observation[np.newaxis, :]})
        #print("prob_weights",prob_weights)
        action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())  # select action w.r.t the actions prob
        return action

    def store_transition(self, s, a, r):
        self.ep_obs.append(s)
        self.ep_as.append(a)
        self.ep_rs.append(r)

    def learn(self):
        # discount and normalize episode reward
        discounted_ep_rs_norm = self._discount_and_norm_rewards()

        # train on episode
        self.sess.run(self.train_op, feed_dict={
             self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
             self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
             self.tf_vt: discounted_ep_rs_norm,  # shape=[None, ]
        })

        self.ep_obs, self.ep_as, self.ep_rs = [], [], []    # empty episode data
        return discounted_ep_rs_norm

    def _discount_and_norm_rewards(self):
        # discount episode rewards
        discounted_ep_rs = np.zeros_like(self.ep_rs)
        running_add = 0
        for t in reversed(range(0, len(self.ep_rs))):
            running_add = running_add * self.gamma + self.ep_rs[t]
            discounted_ep_rs[t] = running_add
        # print("t",t)
        # # normalize episode rewards
        # print("discounted_ep_rs be",discounted_ep_rs)
        # # print("np.mean(discounted_ep_rs)",np.mean(discounted_ep_rs))
        # discounted_ep_rs -= np.mean(discounted_ep_rs)
        # discounted_ep_rs /= np.std(discounted_ep_rs)
        # print("discounted_ep_rs af", discounted_ep_rs)
        return discounted_ep_rs
    def obs_to_state(self,obs):
        state = np.array(obs)
        return state


class PolicyGradientIm:
    def __init__(
            self,
            env,
            n_actions,
            n_features,
            learning_rate=0.01,
            reward_decay=0.95,
            output_graph=False,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay

        self.ep_obs, self.ep_as, self.ep_rs = [], [], []

        self._build_net()
        self.env = env
        self.sess = tf.Session()
        self.getRflag = False
        self.Im = pd.DataFrame(columns=['sta_visul', 'ImS', 'beforestate', 'getstate'],
                               dtype=np.float64)
        if output_graph:
            # $ tensorboard --logdir=logs
            # http://0.0.0.0:6006/
            # tf.train.SummaryWriter soon be deprecated, use following
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())

    def _build_net(self):
        with tf.name_scope('inputs'):
            self.tf_obs = tf.placeholder(tf.float32, [None, self.n_features], name="observations")
            self.tf_acts = tf.placeholder(tf.int32, [None, ], name="actions_num")
            self.tf_vt = tf.placeholder(tf.float32, [None, ], name="actions_value")
        # fc1
        layer = tf.layers.dense(
            inputs=self.tf_obs,
            units=10,
            activation=tf.nn.tanh,  # tanh activation
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            name='fc1'
        )
        # fc2
        all_act = tf.layers.dense(
            inputs=layer,
            units=self.n_actions,
            activation=None,
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            name='fc2'
        )

        self.all_act_prob = tf.nn.softmax(all_act, name='act_prob')  # use softmax to convert to probability

        with tf.name_scope('loss'):
            # to maximize total reward (log_p * R) is to minimize -(log_p * R), and the tf only have minimize(loss)
            neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=all_act, labels=self.tf_acts)   # this is negative log of chosen action
            # or in this way:
            # neg_log_prob = tf.reduce_sum(-tf.log(self.all_act_prob)*tf.one_hot(self.tf_acts, self.n_actions), axis=1)
            loss = tf.reduce_mean(neg_log_prob * self.tf_vt)  # reward guided loss

        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)

    def choose_action(self, observation):
        # print("observation",observation)
        # print("np.newaxis",np.newaxis)
        prob_weights = self.sess.run(self.all_act_prob, feed_dict={self.tf_obs: observation[np.newaxis, :]})
        # print("prob_weights",prob_weights)
        action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())  # select action w.r.t the actions prob
        return action

    def store_transition(self, s, a, r):
        self.ep_obs.append(s)
        self.ep_as.append(a)
        self.ep_rs.append(r)

    def learn(self):
        # discount and normalize episode reward
        discounted_ep_rs_norm = self._discount_and_norm_rewards()

        # train on episode
        self.sess.run(self.train_op, feed_dict={
             self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
             self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
             self.tf_vt: discounted_ep_rs_norm,  # shape=[None, ]
        })

        self.ep_obs, self.ep_as, self.ep_rs = [], [], []    # empty episode data
        return discounted_ep_rs_norm

    def _discount_and_norm_rewards(self):
        # discount episode rewards
        discounted_ep_rs = np.zeros_like(self.ep_rs)
        running_add = 0
        for t in reversed(range(0, len(self.ep_rs))):
            running_add = running_add * self.gamma + self.ep_rs[t]
            discounted_ep_rs[t] = running_add
        # print("t",t)
        # # normalize episode rewards
        # print("discounted_ep_rs be",discounted_ep_rs)
        # # print("np.mean(discounted_ep_rs)",np.mean(discounted_ep_rs))
        # discounted_ep_rs -= np.mean(discounted_ep_rs)
        # discounted_ep_rs /= np.std(discounted_ep_rs)
        # print("discounted_ep_rs af", discounted_ep_rs)
        return discounted_ep_rs
    def obs_to_state(self,obs):
        state = np.array(obs)
        return state
    def stochastic_trj(self,seq):
        """

        :param seq:  Ti

        :return: Ci= DataFrame(index=[sta],columns=['StateCount','StateOrder'], dtype=np.float64)
        """

        s=seq[0]
        sta=str(s) #因为做索引的不能是数组，所以将他变成了字符串
        sta_visul = self.env.state_visualization(s)  # 因为我们要直观的看到最后的结果所以把他转换成可视化的编码
        # print("s",s,"sta",sta,"sta_visual",sta_visul)
        # print("seq.counts ",seq)
        #Ci = pd.DataFrame([{'sta_visul':sta_visul,'StateCount':seq.count(s),'StateOrder':1}],index=[sta],columns=['sta_visul','StateCount','StateOrder'], dtype=np.float64)
        Ci = pd.DataFrame([{'sta_visul': sta_visul,  'StateOrder': 1}], index=[sta],
                          columns=['sta_visul',  'StateOrder'], dtype=np.float64)
        k= 1
        for s in seq:
            sta = str(s)
            if s == 'terminal':
                sta_visul = s
            else:
                sta_visul = self.env.state_visualization(s)  # 因为我们要直观的看到最后的结果所以把他转换成可视化的编码
            if sta not in Ci.index:
                k += 1
                Ci=Ci.append(
                    pd.Series({'sta_visul':sta_visul,'StateCount':seq.count(s),'StateOrder':k},
                              name=sta,
                              )
                )
        return Ci

    def stochastic_trjs(self,trjs):
        """

        :param trjs: 众多轨迹
        'StateCount': 每条轨迹中状态stat出现的次数，Ci（Ci.loc[:,'StateCount']）
        SumWik:only calculate the sum of Wik, where the Wik is the order of state k in the list Ui( Ci.index)
        Pk:the number of S_k in all the trjs
        iPk: 每条轨迹某个状态出现只算一次，看看所有轨迹中有的多少条轨迹包含当前状推
        BEk：the stochastic of states BEfore S_k in each trj
        AFk : the stochastic of states AFter S_k in each trjs
        :return: P， 未顺序排列的统计特性，O，按照出现数量有高到底排列

        """
        seq = trjs[0]
        s=seq[0]

        sta = str(s)  # 因为做索引的不能是数组，所以将他变成了字符串
        sta_visul = self.env.state_visualization(s)  # 因为我们要直观的看到最后的结果所以把他转换成可视化的编码
        iPk = 1
        BEk = 0
        AFk = 0
        P= pd.DataFrame([{'sta_visul':sta_visul,'StateOrder':1,'SumWik':0,'Pk':seq.count(sta),'iPk':iPk,'BEk':BEk,'AFk':AFk,'BEplusAF':(BEk+AFk)}],index=[sta],columns=['sta_visul','StateOrder','SumWik','Pk','iPk','BEk','AFk','BEplusAF'], dtype=np.float64)
        # self.Im = pd.DataFrame([{'ImS':0, 'beforestate':1}],index =[sta] ,columns=['ImS', 'beforestate'],
        #                   dtype=np.float64)
        k=1
        for trj in trjs:
            #每一条轨迹,
            temp = self.stochastic_trj(trj)
            # print("temp \n",temp)
            for StateID in temp.index:
                #每一个状态
                BEk = 0
                AFk = 0
                SO = int(temp.loc[StateID,'StateOrder'])
                if SO == 1:
                    continue#第一个状态不算, 无论之前还是之后
                else:
                    for i in range(1,SO):#本身不算
                        da = temp[temp['StateOrder']==i]
                        BEk = BEk + da.iloc[0,1]# StateCount在第二个
                        #BEk =BEk * i #乘以到达该状态的步数
                    for i in range(SO+1,len(temp)+1):
                        da = temp[temp['StateOrder']==i]
                        AFk = AFk + da.iloc[0,1]# StateCount在第二个
                        #AFk = AFk * i

                if StateID not in P.index:
                    #如果之前没有出现过
                    k+=1
                    P = P.append(
                        pd.Series(
                            {'sta_visul': temp.loc[StateID,'sta_visul'],'StateOrder': k, 'SumWik':temp.loc[StateID,'StateOrder'],'Pk':seq.count(sta),'iPk':iPk,'BEk':BEk,'AFk':AFk,'BEplusAF':(BEk+AFk)},
                            name=StateID,
                        )
                    )
                else:
                    P.loc[StateID,'SumWik'] += temp.loc[StateID,'StateOrder']
                    P.loc[StateID,'Pk'] += temp.loc[StateID,'StateCount']
                    P.loc[StateID,'iPk'] += 1
                    P.loc[StateID,'BEk'] += BEk
                    P.loc[StateID, 'AFk'] += AFk
                    P.loc[StateID, 'BEplusAF'] += (BEk+AFk)
        return P

    def Im_s(self, P, tempn):

        #C = P[P['BEplusAF'] == P['BEplusAF'].max()]  # sort
        C = P.sort_values(by = "BEplusAF",ascending = False)
        #print("max stateID \n", C)
        for StateID in C.index:
            self.check_state_exist_Im(StateID)
            if self.Im.loc[StateID, 'beforestate']>0:
                continue
            else:
                self.Im.loc[StateID, 'sta_visul'] = P.loc[StateID, 'sta_visul']
                self.Im.loc[StateID, 'ImS'] = tempn * (tempn + 1) / 2
                self.Im.loc[StateID, 'beforestate'] =1
                break

        return self.Im

    def check_state_exist_Im(self, state):
        if state not in self.Im.index:
            # append new state to q table
            self.Im = self.Im.append(
                pd.Series(
                    [0]*4,
                    index=self.Im.columns,
                    name=state,
                )
            )



class Actor():
    def __init__(self, sess, n_features, n_actions, lr= 0.001):
        self.sess =sess

        #用placeholder写的都是输入吗？
        self.s = tf.placeholder(tf.float32, [1, n_features], "state")
        self.a = tf.placeholder(tf.int32, None, "act")
        self.td_error = tf.placeholder(tf.float32, None, "td_error")

        with tf.variable_scope('Actor'):
            l1 = tf.layers.dense(
                input=self.s,
                units=20,
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0., .1),
                bias_initializer=tf.constant_initializer(0.1),
                name='l1'
            )
        self.acts_prob = tf.layers.dense(
            input=l1,
            units=n_actions,
            activation=tf.nn.softmax,
            kernel_initializer=tf.random_normal_initializer(0., .1),
            bias_initializer=tf.constant_initializer(0.1),
            name='acts_prob'
        )
        with tf.variable_scope('exp_v'):
            log_prob = tf.log(self.acts_prob[0,self.a])#为啥用log
        self.exp_v = tf.reduce_mean(log_prob*self.td_error) #td error 是加权的作用

        with tf.variable_scope('train'):
            self.train_op =  tf.train.AdamOptimizer(lr).minimize(-self.exp_v)


    def learn(self, s, a, td):
        s = s[np.newaxis,:] #np.newaxis 用来增加维度 为什么要增加维度
        feed_dict={self.s:s, self.a:a, self.td_error:td}
        _,exp_v = self.sess.run([self.train_op,self.exp_v],feed_dict)

        return exp_v


    def choose_action(self,s):
        s = s[np.newaxis, :]  # np.newaxis 用来增加维度
        probs =self.sess.run(self.acts_prob,{self.s:s}) #后面方括号相当于feed dict

        return np.random.choice(np.arange(probs.shape[1]), p= probs.ravel())
    #这里的np.arange 支持步长为小数， ravel 和flatten的作用相同，将多维降为一位



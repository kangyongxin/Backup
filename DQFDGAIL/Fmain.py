# -*- coding: utf-8 -*
'''

#设计对比实验：
#首先要比较相同的示例数据条件下各个算法是否都能完成任务
#1.只有DQFD
#2.只有GAIL with PPO
#3.DQFD+GAIL V4 版本， 基本实现数据通信，但是没有数据反馈的过错

#然后，完善DG， 期望能够得到至少和两个baseline同样的效果
#引入 CGAIL或者 infoGAIL
#引入 vae实现 隐变量存储

#拓展游戏的范围

'''

import tensorflow as tf
import gym
from gym import wrappers
from FConfig import Config
import random
import numpy as np
import pickle
from collections import deque
import itertools
from FMemory import Memory
from FAGENT import AGENT
import matplotlib
import matplotlib.pyplot as plt
from FSOUL import SOUL
from Frun_DQfD import run_DQfD
from Frun_GAIL import run_GAIL
from Frun_MC import run_MC
if __name__ == '__main__':
    env = gym.make(Config.ENV_NAME)
    print("hello world !")
    '''
    # run DQFD  and get scores map
    '''
    # scores_DQfD=run_DQfD(env, Config)
    # plt.plot(scores_DQfD, 'r')
    # plt.show()

    '''
    #run GAIL and get scores map
    
    '''
    #
    # scores_GAIL = run_GAIL(env, Config)
    # plt.plot(scores_GAIL, 'r')
    # plt.show()

    '''
    # run memory_construction
    '''

    scores_MC = run_MC(env, Config)
    plt.plot(scores_MC, 'b')
    plt.show()
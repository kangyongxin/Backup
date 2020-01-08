# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 16:17:21 2019

小环境，希望能找到解，并并完善流程

@author: Administrator
"""

"""
接着State_Im_MAZE.py
获取关键状态，并进行可视化
按步运行，并动态显示我们找到的关键状态

为关键状态赋值，并将其融入Qlearning 算法
设计并完成比较实验，证明， 这种发现关键状态的方法可以加速Q-Learning的收敛

1.隔离状态编码，与，求解过程
2.重新设计策略提升方法
3.从随机策略学起
4.尽量精简流程

"""
import time
import gym
import numpy as np
from baselines.UpTrend.maze_env10 import Maze
from baselines.UpTrend.RL_brain import UptrendVS,QLearningTable, InternalModel
import matplotlib.pyplot as plt
import scipy.signal as signal  # 求极值用
import pandas as pd

def generate_trjs(M_trjs,N_steps,RL,env):
    trjs =[]
    for eps in range(M_trjs):
        trj = []
        step = 0
        observation = env.reset()
        state = RL.obs_to_state(observation)
        trj.append(state)
        while step < N_steps:
            step += 1
            env.render()
            #action = RL.random_action(observation)
            action = RL.choose_action(str(state))
            observation_, reward, done = env.step(action)
            observation = observation_
            if done:
                print("done during generate")
                break
            state = RL.obs_to_state(observation)
            trj.append(state)
        trjs.append(trj)
    return trjs

def show_trjs(trjs,RL):
    for trj in trjs:
        print("trj :")
        for i in range(len(trj)):
            print(RL.show_state(trj[i]))

    # def Im_s1(self,P):
    #     n_state = len(P)
    #     Im = pd.DataFrame([{'ImS': 0, 'beforestate': 0}], index=P.index, columns=['ImS', 'beforestate'],
    #                       dtype=np.float64)
    #
    #     ind_C = Im[P['BEplusAF'] == P['BEplusAF'].max()].index  # argmax??
    #     print(ind_C)
    #     Im.loc[ind_C, 'ImS'] = 1
    #
    #     return Im
# def trainQ(N,RL,env):
#     trainnumber = 100000




def train(N,RL,env,Im):
    """

    :param N: 每次训练的步数
    :return:
    """
    #n_state = len(S_space)
    trainnumber = 100000
    r = np.zeros(trainnumber)
    usingstep = np.zeros(trainnumber)
    steps = 0 # 用来评价收敛的速度

    for eps in range(trainnumber):
        """
        需要重新定义局部收敛的条件，我们只要得到局部的收敛策略即可
        """
        observation = env.reset()
        step = 0
        temp_trj =[]
        while step <N:
            step +=1
            env.render()
            state =  RL.obs_to_state(observation)
            temp_trj.append(state)
            action = RL.choose_action(str(observation))
            observation_, reward, done = env.step(action)
            if reward >0:
                RL.resetIm(reward)
            state_= RL.obs_to_state(observation_)

            if str(state_) in RL.Im.index:
                maxImS = RL.Im['ImS'].max()

                # """ 所有Im中的值都指导探索"""
                # reward = reward*maxImS + maxImS
                # r[eps] = r[eps] + reward  # 用来判断收敛
                # RL.learn(str(state), action, reward, str(state_))
                if RL.Im.loc[str(state_), 'ImS'] == maxImS:

                    """只有最大值指导探索 """
                    reward = reward*maxImS + maxImS
                    r[eps] = r[eps] + reward  # 用来判断收敛
                    RL.learn(str(state), action, reward, str(state_))

                    usingstep[eps] = step
                    print("step", step)
                    break

            r[eps] = r[eps] + reward  # 用来判断收敛
            RL.learn(str(state), action, reward, str(state_))
            observation = observation_


            if done:
                usingstep[eps] = step
                # done=True
                print(step)
                break


        steps = steps + step
        #print("Im \n",RL.Im['ImS'])
        print("max Im \n",RL.Im)
        print("sum r",sum(r[eps - 10:eps]))

        if (sum(r[eps - 10:eps]) > 9* RL.Im['ImS'].max()) and sum(usingstep[eps - 10:eps]) == usingstep[eps] * 10:
            print("this turn have done")
            print("temp_trj",temp_trj)
            for state in temp_trj:
                StateID = str(state)
                RL.check_state_exist_Im(StateID)
                RL.Im.loc[StateID, 'beforestate'] = 1
            break

    return  steps,step


def main_MAZE(env):
    n_state = len(S_space)
    RL = UptrendVS(env,actions=list(range(env.n_actions)))
    M_trjs =30
    N_steps = 20
    N0=5


    tempn =0
    strr =0
    while tempn<100:
        tempn +=1

        trjs = generate_trjs(M_trjs,N_steps,RL,env)
        #show_trjs(trjs,RL)
        #P=RL.stochastic_trj(trjs[1])
        P= RL.stochastic_trjs(trjs)
        print("stochastic value of trjs \n",P)
        #BE = BE_Sk()
        Im = RL.Im_s(P,tempn)
        #print('Im s \n',Im)
        [steps,sr]=train(N_steps+N0, RL, env,Im)
        print("steps",steps)
        N_steps = N_steps + sr




if __name__ == "__main__":
    env = Maze()
    S_space = env.state_space()
    main_MAZE(env)




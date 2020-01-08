# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 16:17:21 2019

from zem

@author: Administrator
"""

"""
接着State_Im_MAZE.py
获取关键状态，并进行可视化
按步运行，并动态显示我们找到的关键状态

为关键状态赋值，并将其融入Qlearning 算法
设计并完成比较实验，证明， 这种发现关键状态的方法可以加速Q-Learning的收敛

"""
import gym
import numpy as np
from baselines.UpTrend.maze_env20 import Maze
from baselines.UpTrend.RL_brain import QLearningTable, InternalModel
import matplotlib.pyplot as plt
import scipy.signal as signal  # 求极值用


def stochastic_trj(seq):
    """

    :param seq:  Ti

    :return: Ui={u_i1, u_i2, ..., u_iW_i} & Ci ={c_i1, c_i2, ...,c_iW_i}
    """
    # print("trj,", seq)
    n_state = len(S_space)
    Ui = []  # list of the unique state, ordered by the first time they appeared
    Ci = []  # list of the number of each state corresponded with Ui
    Wik = np.zeros((n_state, 1))
    k = 0
    for i in seq:
        if Ui.count(i) == 0:
            Ci.append(seq.count(i))  # 按照首次出现的状态，出现的次数
            Ui.append(i)  # 首次出现的状态排序
            k = k + 1
    for k in range(len(S_space)):
        for w in range(len(Ui)):
            if Ui[w] == S_space[k]:
                Wik[k] = w + 1  # 第一个出现的状态计为1， 没有出现的记为0 中,wit为首次出现状态 的次序
    return Ui, Ci, Wik


def State_Importance_calculate(trjs):
    a = []
    r = []
    n_state = len(S_space)
    Im_s = np.zeros((n_state, 1))
    Im_p = np.zeros((n_state, 1))
    for eps in range(len(trjs)):
        # print("trjs",trjs[eps])
        [u_state, count_state] = stochastic_trj(trjs[eps])
        a.append(u_state)
        r.append(count_state)  # 每条轨迹的统计特性。状态出现的次数
        for i in range(len(S_space)):  # 某个状态
            for w in range(len(a[eps])):  # 第w个 出现
                if a[eps][w] == S_space[i]:
                    Im_s[i] = Im_s[i] + w  # 出现的顺序 越晚月重要？轨迹中
                    Im_p[i] = Im_p[i] + r[eps][w]  # 状态出现的次数
    for i in range(len(S_space)):
        Im_s[i] = Im_p[i] / Im_s[i]
        if Im_s[i] < 0.01:
            Im_s[i] = 0
    return Im_s


def Im_SumWik(trjs):
    """
    only calculate the sum of Wik, where the Wik is the order of state k in the list Ui
    :param trjs:
    :return:
    """
    U = []
    C = []
    n_state = len(S_space)
    SumWi = np.zeros((n_state, 1))
    for eps in range(len(trjs)):
        [Ui, Ci, Wik] = stochastic_trj(trjs[eps])
        SumWi = SumWi + Wik
    return SumWi


def Im_Pk(trjs):
    """
    the number of S_k in all the trjs
    :param trjs:
    :return: Im_s
    """
    n_state = len(S_space)
    P = np.ones((n_state, 1))
    P2 = np.ones((n_state, 1))
    for eps in range(len(trjs)):
        [Ui, Ci, Wik] = stochastic_trj(trjs[eps])
        for k in range(len(S_space)):
            if Wik[k] == 0:  # 没有出现的不计
                continue
            else:
                ind = int(Wik[k]) - 1  # 第一个出现的状态计为1， 没有出现的记为0
                P[k] = P[k] + Ci[ind]
                P2[k] = P2[k] + 1
    return P, P2


def Im_BEk(trjs):
    """
    the stochastic of states BEfore S_k in each trj
    :param trjs:
    :return:
    """
    n_state = len(S_space)
    BE = np.zeros((n_state, 1))
    for eps in range(len(trjs)):
        [Ui, Ci, Wik] = stochastic_trj(trjs[eps])
        for k in range(len(S_space)):
            if Wik[k] == 0 or Wik[k] == 1:  # 没有出现的和第一个出现的都不计入
                continue  # never appear in this trj
            else:
                ind = int(Wik[k]) - 1
                for j in range(ind):  # 只计算之前的，自己不算
                    BE[k] = BE[k] + Ci[j]
    return BE


def Im_AFk(trjs):
    """
    the stochastic of states AFter S_k in each trjs
    :param trjs:
    :return:
    """
    n_state = len(S_space)
    AF = np.zeros((n_state, 1))
    for eps in range(len(trjs)):
        [Ui, Ci, Wik] = stochastic_trj(trjs[eps])
        for k in range(len(S_space)):
            if Wik[k] == 0 or Wik[k] == len(Ui):  # 没有出现的和第一个出现的都不计入
                continue  # never appear in this trj
            else:
                ind = int(Wik[k])
                for j in range(ind + 1, len(Ui)):  # 只计算之前的，自己不算
                    AF[k] = AF[k] + Ci[j]
    AF[0] = 0
    return AF


def Im_O_OBE_OAF(trjs):
    """
    Order Pk, BEk, and AFk
    :param trjs:
    :return: O OBE AF
    """
    n_state = len(S_space)
    [P, P2] = Im_Pk(trjs)

    BE = Im_BEk(trjs)
    AF = Im_AFk(trjs)
    O = np.sort(-P, axis=0)  # 下降
    index_O = np.argsort(-P, axis=0)
    OBE = np.zeros((n_state, 1))
    OAF = np.zeros((n_state, 1))
    i = 0
    for ind in index_O:
        OBE[i] = BE[ind]
        OAF[i] = AF[ind]
        i += 1
    return -O, index_O, OBE, OAF, P, P2, BE, AF


def generate_trjs(n_trj, N, RL, env):
    trjs = []
    for eps in range(n_trj):
        observation = env.reset()
        state = env.obs_to_state(observation)
        trj = []
        trj.append(state)
        step = 0

        while step < N:
            step += 1
            env.render()
            # action = RL.random_action(str(observation))
            action = RL.choose_action(str(observation))
            observation_, reward, done = env.step(action)
            if reward == -1:
                break
            observation = observation_
            if done:
                print("done during generate!")
                N = N - 3
                break
            state = env.obs_to_state(observation)
            # print(state)

            trj.append(state)
        trjs.append(trj)
    return trjs


def Im_s(trjs, beforestate):
    n_state = len(S_space)
    Im = np.zeros((n_state, 1))
    O, index_O, OBE, OAF, P, P2, BE, AF = Im_O_OBE_OAF(trjs)
    C = AF+ BE
    """
    show OBE VS    indexo 拍完的索引    O 简单到难    P123456正序
    """
    # txt_O = index_O.copy() + 1
    # firms = txt_O
    # x_axs = [i for i in range(len(txt_O))]
    # ax4 = plt.subplot(1, 1, 1)
    # plt.plot(C, color='red', label="point")
    # # plt.xticks(x_axs,range(1,37) , rotation=90)
    # plt.xticks(x_axs, range(1, 37), rotation=90)
    # ax4.set_title("=AF/P2+BE/P2")
    # plt.show()
    """
    calculate Im
    """

    maxc = 0
    for i in range(400):
        if C[i] > maxc and beforestate[i] == 0:
            k = i + 1
            maxc = C[i]

    inds, _ = signal.argrelextrema(C, np.greater)
    # print(inds)
    # print("MAX POINT OF OBE ", inds) inds 极值索引
    # l= len(inds)
    # for i in range(l):
    # Im[index_O[inds[i]]]= 0*(i+1)/l # 对应回到原先的状态上
    return k
def resetIm(Im):
    print("we need to reset Im")
    return Im

def train(N, RL, env, Im, turns):
    trainnumber = 1000000
    r = np.zeros(trainnumber)
    usingstep = np.zeros(trainnumber)

    for eps in range(trainnumber):
        """
        需要重新定义局部收敛的条件，我们只要得到局部的收敛策略即可
        """
        getstatelogo = np.zeros(400)
        beforestate = np.zeros(400)
        observation = env.reset()
        state = env.obs_to_state(observation)
        step = 0  # 规定步骤内，稳定地达到某个分数就为局部收敛

        while step < N:
            step += 1
            env.render()

            # action = RL.random_action(str(observation))
            action = RL.choose_action(str(observation))
            observation_, reward, done = env.step(action)
            if reward >0:
                Im=resetIm(Im)
            if done:
                state = 400
            else:
                state = env.obs_to_state(observation_)
            # print(state)
            beforestate[state - 1] = 1
            if Im[state - 1] > 0 and getstatelogo[state - 1] == 0:
                reward = reward + Im[state - 1] * 0.1
                getstatelogo[state - 1] = getstatelogo[state - 1] + 1
            # print(state,step)
            r[eps] = r[eps] + reward
            RL.learn(str(observation), action, reward, str(observation_))
            observation = observation_

            # print(Im)
            if Im[state - 1] == max(Im):
                usingstep[eps] = step
                # done=True
                print(step)
                break
                # print(done,step)

        # if eps>9:
        # print(sum(r[eps-10:eps]),sum(usingstep[eps-10:eps]))
        if (sum(r[eps - 10:eps]) > 0.9 * sum(Im) and sum(usingstep[eps - 10:eps]) == usingstep[eps] * 10):
            print("this turn have done")
            break

    return beforestate, step




def main_MAZE(env):
    n_trj = 100
    N = 50
    # Delta = 5
    Im = np.zeros(400)
    RL = QLearningTable(actions=list(range(env.n_actions)))
    tempn = 0
    strr = 0
    beforestate = np.zeros(400)
    while tempn < 100:
        tempn += 1
        trjs = generate_trjs(n_trj, N, RL, env)  # 用现有策略产生轨迹
        Im[Im_s(trjs, beforestate) - 1] = tempn  # 根据目前轨迹，计算每个状态的价值
        print("Im", Im)

        # Im[tempn-1]=1
        [beforestate, strr] = train(N, RL, env, Im, tempn)
        print('neesstep', strr)  # 根据各个状态的价值提升策略
        N = N + strr  # 延长探索步骤


if __name__ == "__main__":
    env = Maze()
    S_space = env.state_space()
    main_MAZE(env)
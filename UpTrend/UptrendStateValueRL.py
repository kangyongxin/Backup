"""
接着State_Im_MAZE.py
获取关键状态，并进行可视化
按步运行，并动态显示我们找到的关键状态

为关键状态赋值，并将其融入Qlearning 算法
设计并完成比较实验，证明， 这种发现关键状态的方法可以加速Q-Learning的收敛

"""
import gym
import numpy as np
from baselines.UpTrend.maze_env_noend import Maze
from baselines.UpTrend.RL_brain import QLearningTable,InternalModel
import matplotlib.pyplot as plt
import scipy.signal as signal #求极值用

def stochastic_trj(seq):
    """

    :param seq:  Ti

    :return: Ui={u_i1, u_i2, ..., u_iW_i} & Ci ={c_i1, c_i2, ...,c_iW_i}
    """
    #print("trj,", seq)
    n_state = len(S_space)
    Ui = []  # list of the unique state, ordered by the first time they appeared
    Ci = []  # list of the number of each state corresponded with Ui
    Wik = np.zeros((n_state, 1))
    k = 0
    for i in seq:
        if Ui.count(i) == 0:
            Ci.append(seq.count(i))
            Ui.append(i)
            k = k + 1
    for k in range(len(S_space)):
        for w in range(len(Ui)):
            if Ui[w]== S_space[k]:
                Wik[k]=w+1 #第一个出现的状态计为1， 没有出现的记为0
    return Ui, Ci, Wik

def State_Importance_calculate(trjs):
    a=[]
    r=[]
    n_state= len(S_space)
    Im_s = np.zeros((n_state, 1))
    Im_p = np.zeros((n_state, 1))
    for eps in range(len(trjs)):
        #print("trjs",trjs[eps])
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
        if Im_s[i]<0.01:
             Im_s[i]=0
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
        SumWi=SumWi + Wik
    return SumWi

def Im_Pk(trjs):
    """
    the number of S_k in all the trjs
    :param trjs:
    :return: Im_s
    """
    n_state = len(S_space)
    P = np.zeros((n_state, 1))
    for eps in range(len(trjs)):
        [Ui, Ci, Wik] = stochastic_trj(trjs[eps])
        for k in range(len(S_space)):
            if Wik[k]==0:#没有出现的不计
                continue
            else:
                ind= int(Wik[k])-1#第一个出现的状态计为1， 没有出现的记为0
                P[k]=P[k]+Ci[ind]
    return P

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
            if Wik[k] == 0 or Wik[k] == 1:#没有出现的和第一个出现的都不计入
                continue # never appear in this trj
            else:
                ind=int(Wik[k])-1
                for j in range(ind): #只计算之前的，自己不算
                    BE[k]=BE[k]+Ci[j]
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
                ind = int(Wik[k]) - 1
                for j in range(ind+1,len(Ui)):  # 只计算之前的，自己不算
                    AF[k] = AF[k] + Ci[j]
    AF[0]=0
    return AF
def Im_O_OBE_OAF(trjs):
    """
    Order Pk, BEk, and AFk
    :param trjs:
    :return: O OBE AF
    """
    n_state = len(S_space)
    P = Im_Pk(trjs)
    BE = Im_BEk(trjs)
    AF = Im_AFk(trjs)
    O = np.sort(-P,axis=0)# 下降
    index_O = np.argsort(-P,axis=0)
    OBE = np.zeros((n_state, 1))
    OAF = np.zeros((n_state, 1))
    i=0
    for ind in index_O:
        OBE[i] = BE[ind]
        OAF[i] = AF[ind]
        i+=1
    return -O,index_O,OBE,OAF,P,BE,AF

def generate_trjs(n_trj,N,RL,env):

    trjs = []
    for eps in range(n_trj):
        print("generating trjs")
        observation = env.reset()
        # state = env.obs_to_state(observation)
        trj = []
        #trj.append(state)
        step = 0

        while step < N:
            step += 1

            state = env.obs_to_state(observation)
            trj.append(state)
            #action = RL.random_action(str(observation))
            action = RL.choose_action(str(observation))
            observation_, reward, done = env.step(action)
            observation = observation_
            env.render()

            if done:
                print("done during generate!")
                # N = N - 3
                break

        trjs.append(trj)
    return trjs

def Im_s(trjs):
    n_state = len(S_space)
    Im = np.zeros((n_state, 1))
    O, index_O, OBE, OAF, P, BE, AF = Im_O_OBE_OAF(trjs)

    """
    show OBE VS indexo
    """
    # txt_O = index_O.copy() + 1
    # firms = txt_O
    # x_axs = [i for i in range(len(txt_O))]
    # ax4 = plt.subplot(1, 1, 1)
    # plt.plot(OBE, color='red',label="point")
    # plt.xticks(x_axs, firms, rotation=90)
    # ax4.set_title("OBE")
    # plt.show()
    """
    calculate Im
    """

    #inds,_ = signal.argrelextrema(OBE, np.greater)
    MAX = 0
    ind = 0
    C=OBE
    print("C",C)
    for i in range(len(C)):
        if C[i] >MAX:
            MAX = C[i]
            ind = i
    print("temp oval: ", index_O[ind]+1)
    Im[index_O[ind]] = 1
    # MAX = 0
    # ind = 0
    # print("OAF",OAF)
    # print("index_O",index_O)
    # for i in range(len(OAF)):
    #     if OAF[i] > MAX:
    #         MAX = OAF[i]
    #         ind = i
    # print("temp oval: ", index_O[ind] + 1)
    # Im[index_O[ind]] = 1
    print("ind",ind)
    print("Im",Im)
    # print(OBE)
    # inds = OBE.index(max(OBE))
    #print("MAX POINT OF OBE ", inds)
    # l= len(inds)
    # for i in range(l):
    #     Im[index_O[inds[i]]]=1 # 对应回到原先的状态上
    #     print("temp oval: ",index_O[inds[i]] )
    return Im
def train(n_trj,N,RL,env,Im):
    steps = []
    count =0
    N_Delta = 100
    while True:
        """
        需要重新定义局部收敛的条件，我们只要得到局部的收敛策略即可
        """
        observation = env.reset()
        state = env.obs_to_state(observation)
        step = 0 #规定步骤内，稳定地达到某个分数就为局部收敛

        while True:
            step += 1
            env.render()
            # action = RL.random_action(str(observation))
            action = RL.choose_action(str(observation))
            observation_, reward, done = env.step(action)
            if reward == -1:
                break
            #print("observation_",observation_)
            state = env.obs_to_state(observation_)
            reward = Im[state - 1] + reward

            RL.learn(str(observation), action, reward, str(observation_))
            observation = observation_
            if done or reward==1:
                print("done steps: ", step)
                break
        # print("step",step)
        steps.append(step)
        print("step",step)
        if len(steps) <10:
            N_Delta = steps[-1]
            continue
        if step< N_Delta:
            N_Delta = step
        elif step == N_Delta:
            count +=1
            print(count)
            if count > 5:
                break
            else:
                continue
        elif step > N_Delta:
            count =0
        print("N_Delta", N_Delta)
    # print("steps",steps)
    # plt.plot(steps)
    # plt.show()
    # plt.close()

    return N_Delta









def main_MAZE(env):
    n_trj = 1000
    N = 10

    RL = QLearningTable(actions=list(range(env.n_actions)))
    tempn = 0
    while tempn<10:
        tempn+=1
        print("tempn",tempn)
        trjs=generate_trjs(n_trj,N,RL,env) #用现有策略产生轨迹
        Im = Im_s(trjs)#根据目前轨迹，计算每个状态的价值
        #print("Im",Im)
        N_Delta = train(n_trj,N,RL,env,Im)#根据各个状态的价值提升策略
        N = N+ N_Delta #延长探索步骤






if __name__ == "__main__":
    env = Maze()
    S_space = env.state_space()
    main_MAZE(env)
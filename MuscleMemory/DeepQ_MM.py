"""
复现Dyna-Q算法， 环境是之前的maze
参考 ： https://github.com/thomas861205/RL-HW3

1.实现基本 的 one step  q learning
2. 实现一个记录状态动作对儿的模型
3. planning 引入到 q表的 更新中

"""

"""
Qlearning 的基本步骤：

1 初始化 q表
2 观测值
3 用 greedy方法采样 动作
4 实施动作 得到 反馈
5 用反馈值 和状态动作一起 更新q表
6 循环

"""
"""
需要外加的部分：
model 要初始化
model 每次有新的值就要更新一下，（这里其实只是把见到的状态记录一下而已，并没有训练什么函数，是个确定性模型）
然后每次迭代之后，都要有n次planning的过程，n是我们要比较的量
"""
from baselines.MuscleMemory.maze_env import Maze
from baselines.MuscleMemory.RL_brain import QLearningTable,InternalModel
import matplotlib.pyplot as plt
import math
import numpy as np
from sklearn import preprocessing

"""
obs are something we see
states are something we know
state is the hidden variable
"""
def obs_to_state(obs):
    """
    obs=[45.0, 5.0, 75.0, 35.0]
    第1行 第2列
    state = 2
    state:
    1 2 3 4 5
    6 7 8 9 10
    ...

    """
    states= ((obs[1]+15.0 - 20.0)/40)*5 + (obs[0] +15.0 - 20.0)/40 +1
    return int(states)

"""
stochastic characters of trj
先后经历的状态，以及相应状态的个数
"""
def stochastic_trj(seq):
    print("trj,", seq)
    u_state = []  # list of the unique state
    count_state = []  # list of the number of state
    k = 0
    for i in seq:
        if u_state.count(i) == 0:
            count_state.append(seq.count(i))
            u_state.append(i)
            k = k + 1
    print("u_s", u_state)
    print("count_s", count_state)

    return u_state, count_state

def MaxMinNormalization(x):
    Min = np.min(x)
    Max = np.max(x)
    x = (x - Min) / (Max - Min)
    return x
def MaxNormalization(x):
    Max = np.max(x)
    for i in range(len(x)):
        if x[i] == Max:
            x[i]=1
        else:
            x[i]=0
    return x

def Max2Normalization(x):
    #print("before sort:", x)
    y=np.zeros((x.shape[0], 1))
    Maxsort=sorted(x)
    print("after sort:",Maxsort)
    for i in range(len(x)):
        if x[i] == Maxsort[-1]:
            y[i]=1
        else:
            if x[i]== Maxsort[-2]:
                y[i] =0.1
            else:
                if x[i]==Maxsort[-3]:
                    y[i] = 0.01
                else:
                    y[i]=0
    return y

def update():
    steps_plot = []
    a =[]
    e=[]
    r =[]
    oo = []
    for i in range(25):
        oo.append(i)
    n_state= len(oo)
    n_trj= 100
    Im_s = np.zeros((n_state, 1))
    Norm_Ims = np.zeros((n_state, 1))
    for episode in range(n_trj):
        # initial observation
        observation = env.reset()

        steps = 0
        tao=0
        trj = []

        while steps<200:
            # fresh env
            state = obs_to_state(observation)
            trj.append(state)
            env.render()
            steps= steps+1
            tao=tao+1
            # RL choose action based on observation
            action = RL.choose_action(str(observation))
            #action = RL.random_action(str(observation))
            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)

            # RL learn from this transition
            state_ = obs_to_state(observation)
            if state_ in trj:
                reward = reward #we only only add the importance of the state at the first time we met it
            else:
                reward = reward + fac*Norm_Ims[state_-1]# be affinitive with the coding algorithm


            RL.learn(str(observation), action, reward, str(observation_))

            # 根新模型
            model.store(observation, action, observation_, reward)
            # swap observation
            observation = observation_
            """ planning n """
            for n in range(plan_n):
                (stateP, actionP, next_stateP, rewardP) = model.sample()#这里是随机采的，但是可以与当前状态相关地采吗？
                RL.learn(str(stateP), actionP, rewardP, str(next_stateP))

            # break while loop when end of this episode
            if done:
                if reward == -1:
                    steps = 200
                break
        steps_plot.append(steps)
        #_____________________________________
        #Calculate the Importance of S after each episode
        # Im_s =
        #
        [u_state, count_state] = stochastic_trj(trj)

        a.append(u_state)
        e.append(trj)
        r.append(count_state)#每条轨迹的统计特性。状态出现的次数
        for i in range(len(oo)):  # 某个状态
            for w in range(len(a[episode])):  # 第w个 出现
                if a[episode][w] == oo[i]:
                    Im_s[i] = Im_s[i] + w  # 出现的顺序 越晚月重要？轨迹中

        print("Im_s",Im_s.T)
        #normalizer = preprocessing.Normalizer().fit(Im_s)
        #Norm_Ims=MaxNormalization(Im_s)
        Norm_Ims = Max2Normalization(Im_s)
        print("Norm_Ims",Norm_Ims.T)
    # print("oo", oo)
    # print("p", p)
    # print("u", u)
    # print("uo", p / uo)
    # temp= uo
    # # # plt.plot(oo,p)
    # # # plt.show()
    # # plt.plot(oo, p / uo)
    # # plt.show()
    # # 第一行第一列图形
    # ax1 = plt.subplot(3, 1, 1)
    # # 第一行第二列图形
    # ax2 = plt.subplot(3, 1, 2)
    # # 第二行
    # ax3 = plt.subplot(3, 1, 3)
    #
    # plt.sca(ax1)
    # # 绘制红色曲线
    # plt.plot(oo, p, color='red',marker='*')
    # plt.ylabel('p')
    #
    # plt.sca(ax2)
    # # 绘制蓝色曲线
    # plt.plot(oo, u, color='blue',marker='*')
    # plt.ylabel('u')
    #
    # plt.sca(ax3)
    # # 绘制绿色曲线
    # plt.plot(oo, temp, color='green',marker='*')
    # plt.ylabel('uo')
    #
    # plt.show()

    plt.plot(steps_plot, label="deepq_mm")
    plt.show()
    # end of game
    print('game over')
    env.destroy()

if __name__ == "__main__":
    env = Maze()
    dynaQplus= False
    MuscleMemory = True
    if dynaQplus:
        kappa=0.00001
    if MuscleMemory:
        fac = 0.001
    RL = QLearningTable(actions=list(range(env.n_actions)))
    model = InternalModel()
    plan_n = 0
    env.after(100, update)
    env.mainloop()
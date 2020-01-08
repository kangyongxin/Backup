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
    for episode in range(n_trj):
        # initial observation
        observation = env.reset()

        steps = 0
        tao=0
        trj = []
        while True:
            # fresh env
            state = obs_to_state(observation)
            trj.append(state)
            #env.render()
            steps= steps+1
            tao=tao+1
            # RL choose action based on observation
            action = RL.choose_action(str(observation))
            #action = RL.random_action(str(observation))
            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)
            if dynaQplus :
                if model.check(observation, action, observation_, reward):
                    tao=tao-1
                reward = reward - kappa*math.sqrt(tao)
            # RL learn from this transition
            #print(reward)
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
                break
        steps_plot.append(steps)
        #________________________________________
        #
        [u_state, count_state] = stochastic_trj(trj)

        a.append(u_state)
        e.append(trj)
        r.append(count_state)#每条轨迹的统计特性。状态出现的次数
    p = np.zeros((n_state, 1))
    u = np.zeros((n_state, 1))
    uo = np.zeros((n_state, 1))
    for j in range(n_trj):  # 第j条轨迹
        for i in range(len(oo)):  # 某个状态
            for w in range(len(a[j])):  # 第w个 出现
                if a[j][w] == oo[i]:
                    p[i] = p[i] + sum(r[j][0:w])  # 第j条轨迹中第i个状态出现之前所有的状态频率统计
                    u[i] = u[i] + w  # 出现的顺序 越晚月重要？轨迹中
                    uo[i] = uo[i] + r[j][w]  # 第 i 个状态在所有轨迹中出现的次数总和
    print("oo", oo)
    print("p", p)
    print("u", u)
    print("uo", p / uo)
    temp= uo
    # # plt.plot(oo,p)
    # # plt.show()
    # plt.plot(oo, p / uo)
    # plt.show()
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

    plt.plot(steps_plot, label='%d steps planning ahead' % (plan_n))
    plt.show()
    # end of game
    print('game over')
    env.destroy()

if __name__ == "__main__":
    env = Maze()
    dynaQplus= False
    if dynaQplus:
        kappa=0.00001
    RL = QLearningTable(actions=list(range(env.n_actions)))
    model = InternalModel()
    plan_n = 50
    env.after(100, update)
    env.mainloop()
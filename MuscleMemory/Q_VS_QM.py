"""

here, we just evaluate the average steps to demonstrate that our method , named Qmm, can achieve the goal steadily
in less steps than Q_leaning .
"""

"""

"""
from baselines.MuscleMemory.maze_env import Maze
from baselines.MuscleMemory.RL_brain import QLearningTable,InternalModel
import matplotlib.pyplot as plt
import math
import numpy as np
from sklearn import preprocessing
import pandas
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
    states= ((obs[1]+15.0 - 20.0)/40)*6 + (obs[0] +15.0 - 20.0)/40 +1
    return int(states)

def stochastic_trj(seq):
    #print("trj,", seq)
    u_state = []  # list of the unique state
    count_state = []  # list of the number of state
    k = 0
    for i in seq:
        if u_state.count(i) == 0:
            count_state.append(seq.count(i))
            u_state.append(i)
            k = k + 1
    #print("u_s", u_state)
    #print("count_s", count_state)

    return u_state, count_state

def Max3Normalization(x):
    #print("before sort:", x)
    y=np.zeros((x.shape[0], 1))
    Maxsort=sorted(x)
    #print("after sort:",Maxsort)
    for i in range(len(x)):
        if x[i] == Maxsort[-1]:
            y[i]=0.55
        else:
            if x[i]== Maxsort[-2]:
                y[i] =0.5
            else:
                if x[i]==Maxsort[-3]:
                    y[i] = 0.45
                else:
                    y[i]=0
    return y
def q_learning():
    steps_plot = []
    RL = QLearningTable(actions=list(range(env.n_actions)))
    n_trj = 400
    for episode in range(n_trj):
        # initial observation
        observation = env.reset()

        steps = 0
        while steps < 50:
            # fresh env
            # env.render()
            steps = steps + 1
            # RL choose action based on observation
            action = RL.choose_action(str(observation))
            # action = RL.random_action(str(observation))
            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)

            RL.learn(str(observation), action, reward, str(observation_))
            # swap observation
            observation = observation_

            # break while loop when end of this episode
            if done:
                if reward == -1:
                    steps = 50
                break
        steps_plot.append(steps)
    return steps_plot

def q_mm_learning():
    steps_plot = []
    RL = QLearningTable(actions=list(range(env.n_actions)))
    a = []
    e = []
    r = []
    oo = []
    for i in range(36):
        oo.append(i)
    n_state = len(oo)
    n_trj = 400
    fac = 0.1
    Im_s = np.zeros((n_state, 1))
    Norm_Ims = np.zeros((n_state, 1))
    for eps in range(800):
        print(eps)
        observation = env.reset()
        trj = []
        while True:
            state = obs_to_state(observation)
            trj.append(state)
            action = RL.random_action(str(observation))
            observation_, reward, done = env.step(action)
            observation = observation_
            if done:
                break
        [u_state, count_state] = stochastic_trj(trj)
        a.append(u_state)
        e.append(trj)
        r.append(count_state)  # 每条轨迹的统计特性。状态出现的次数
        for i in range(len(oo)):  # 某个状态
            for w in range(len(a[eps])):  # 第w个 出现
                if a[eps][w] == oo[i]:
                    Im_s[i] = Im_s[i] + w  # 出现的顺序 越晚月重要？轨迹中
    # plt.plot(Im_s)
    # plt.savefig('./pre_state_value.png')
    for episode in range(n_trj):
        # initial observation
        observation = env.reset()

        steps = 0
        trj = []

        while steps < 50:
            # fresh env
            state = obs_to_state(observation)
            trj.append(state)
            #env.render()
            steps = steps + 1
            # RL choose action based on observation
            if steps > 0:
                action = RL.choose_action(str(observation))
            else:
                action = RL.random_action(str(observation))
            # action = RL.random_action(str(observation))
            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)

            # RL learn from this transition
            state_ = obs_to_state(observation)
            if state_ in trj:
                reward = reward  # we only only add the importance of the state at the first time we met it
            else:
                reward = reward + fac * Norm_Ims[state_ - 1]  # be affinitive with the coding algorithm

            RL.learn(str(observation), action, reward, str(observation_))

            # swap observation
            observation = observation_

            # break while loop when end of this episode
            if done:
                if reward == -1:
                    steps = 50
                fac = steps * fac / 50  # when we got the real reward from the env , the state value should be weaken
                break
        steps_plot.append(steps)
        # _____________________________________
        # Calculate the Importance of S after each episode
        # Im_s =
        #
        [u_state, count_state] = stochastic_trj(trj)

        a.append(u_state)
        e.append(trj)
        r.append(count_state)  # 每条轨迹的统计特性。状态出现的次数
        for i in range(len(oo)):  # 某个状态
            for w in range(len(a[episode])):  # 第w个 出现
                if a[episode][w] == oo[i]:
                    Im_s[i] = Im_s[i] + w  # 出现的顺序 越晚月重要？轨迹中


        # normalizer = preprocessing.Normalizer().fit(Im_s)
        # Norm_Ims=MaxNormalization(Im_s)
        Norm_Ims = Max3Normalization(Im_s)


    return steps_plot

def update():
    #------Q learning----------#
    Qlist=[]
    for t in range(10):
        steps_Q = q_learning()
        Qlist.append(steps_Q)
    print("Qlist",Qlist)
    np.savetxt("Qlist0625-4.txt", Qlist)
    average_q = np.mean(Qlist,axis=0)
    print("average_q",average_q)

    #------Q MM ---------------#
    Qmm_list=[]
    for t in range(10):
        steps_Qmm = q_mm_learning()
        Qmm_list.append(steps_Qmm)
    print("Qmm_list",Qmm_list)
    np.savetxt("Qmm_list0625-4.txt", Qmm_list)
    average_qmm= np.mean(Qmm_list, axis=0)
    print("average_qmm",average_qmm)

    #plt.subplot(2, 1, 1)
    plt.plot(average_q, color='blue')
    #plt.subplot(2, 1, 2)
    plt.plot(average_qmm,color='red')

    plt.show()

if __name__ == "__main__":
    env = Maze()
    env.after(100, update)
    env.mainloop()
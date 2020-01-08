"""

To build a baseline(deep q learning ) and visulize it

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


def update():
    steps_plot = []
    n_trj= 100
    for episode in range(n_trj):
        # initial observation
        observation = env.reset()

        steps = 0
        while True:
            # fresh env
            #env.render()
            steps= steps+1
            # RL choose action based on observation
            action = RL.choose_action(str(observation))
            #action = RL.random_action(str(observation))
            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)

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

    #display
    plt.plot(steps_plot, label='%d steps planning ahead' % (plan_n))
    plt.show()
    # end of game
    print('game over')
    env.destroy()

if __name__ == "__main__":
    env = Maze()
    plan_n = 0
    RL = QLearningTable(actions=list(range(env.n_actions)))
    model = InternalModel()

    env.after(100, update)
    env.mainloop()
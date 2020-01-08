"""
Policy Gradient, Reinforcement Learning.

The cart pole example

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
Tensorflow: 1.0
gym: 0.8.0
"""

import gym
from baselines.UpTrend.RL_brain6 import PolicyGradient,PolicyGradientIm
import matplotlib.pyplot as plt
from baselines.UpTrend.maze_env6 import Maze

DISPLAY_REWARD_THRESHOLD = -2000  # renders environment if total episode reward is greater then this threshold
# episode: 154   reward: -10667
# episode: 387   reward: -2009
# episode: 489   reward: -1006
# episode: 628   reward: -502

RENDER = False  # rendering wastes time

# env = gym.make('MountainCar-v0')
env = Maze()
#env.seed(1)     # reproducible, general Policy gradient has high variance 好像没有这个函数
#env = env.unwrapped
#env = Maze()
# print(env.action_space)
# print(env.observation_space)
# print(env.observation_space.high)
# print(env.observation_space.low)
''''
rl policy g
'''
# RL = PolicyGradient(
#     n_actions=env.action_space.n,
#     n_features=env.observation_space.shape[0],
#     learning_rate=0.02,
#     reward_decay=0.995,
#     # output_graph=True,
# )
# RL = PolicyGradient(
#     n_actions=4,
#     n_features=4,
#     learning_rate=0.11,
#     reward_decay=0.95,
#     # output_graph=True,
# )
#
# for i_episode in range(1000):
#
#     observation = env.reset()
#     step = 0
#     while step<2000:
#         step =step +1
#         if RENDER: env.render()
#         state = RL.obs_to_state(observation)#
#         #action = RL.choose_action(observation)
#         action = RL.choose_action(state)
#
#         observation_, reward, done= env.step(action)     # reward = -1 in all cases
#         reward =float(reward) # 回报要是float 格式
#         #print("observation",observation)
#         env.render()
#         RL.store_transition(observation, action, reward)
#
#         if done:
#             # calculate running reward
#             ep_rs_sum = sum(RL.ep_rs)
#             #print("RL.ep_rs",RL.ep_rs)
#             if 'running_reward' not in globals():
#                 running_reward = ep_rs_sum
#             else:
#                 running_reward = running_reward * 0.99 + ep_rs_sum * 0.01
#             if running_reward > DISPLAY_REWARD_THRESHOLD: RENDER = True     # rendering
#
#             print("episode:", i_episode, "  reward:", int(running_reward),"step",step)
#
#             vt = RL.learn()  # train
#
#             if i_episode == 100:
#                 plt.plot(vt)  # plot the episode vt
#                 plt.xlabel('episode steps')
#                 plt.ylabel('normalized state-action value')
#                 plt.show()
#
#             break
#
#         observation = observation_

'''
new rl

'''



def generate_trjs(M_trjs,N_steps,RL,env):

    trjs =[]
    for eps in range(M_trjs):
        trj = []
        step = 0
        observation = env.reset()
        state = RL.obs_to_state(observation)
        trj.append(observation)
        while step<N_steps:#RL.getRflag == False:
            step += 1
            env.render()
            #action = RL.random_action(observation)
            action = RL.choose_action(state)
            observation_, reward, done = env.step(action)
            observation = observation_
            if done:
                # print("done during generate")
                # RL.getRflag = True
                # state=RL.obs_to_state(observation)
                # trj.append(state)
                # RL.resetIm(trj,reward)
                break
            state = RL.obs_to_state(observation)
            trj.append(observation)
        trjs.append(trj)
    return trjs



RL = PolicyGradientIm(
    env= env,
    n_actions=4,
    n_features=4,
    learning_rate=0.11,
    reward_decay=0.95,
    # output_graph=True,
)

M_trjs = 20  # 30
N_steps = 10  # 10
tempn=1
trjs = generate_trjs(M_trjs, N_steps , RL, env)
#print(trjs)
P= RL.stochastic_trjs(trjs)
print("P \n",P)
Im = RL.Im_s(P,tempn)
print("Im \n", Im)
for i_episode in range(1000):
    trj = []
    observation = env.reset()
    step = 0
    trj.append(observation)
    while step<200:
        step =step +1
        if RENDER: env.render()
        state = RL.obs_to_state(observation)#
        #action = RL.choose_action(observation)
        action = RL.choose_action(state)
#
        observation_, reward, done= env.step(action)     # reward = -1 in all cases
        reward =float(reward) # 回报要是float 格式
#         #print("observation",observation)
        env.render()
        if str(str(observation_)) in RL.Im.index:
            reward = reward + RL.Im.loc[str(observation_),'ImS']

#
        if done:
            # calculate running reward
            ep_rs_sum = sum(RL.ep_rs)
            #print("RL.ep_rs",RL.ep_rs)
            if 'running_reward' not in globals():
                running_reward = ep_rs_sum
            else:
                running_reward = running_reward * 0.99 + ep_rs_sum * 0.01
            if running_reward > DISPLAY_REWARD_THRESHOLD: RENDER = True     # rendering
#
            print("episode:", i_episode, "  reward:", int(running_reward),"step",step)
#
            vt = RL.learn()  # train
#
#             if i_episode == 100:
#                 plt.plot(vt)  # plot the episode vt
#                 plt.xlabel('episode steps')
#                 plt.ylabel('normalized state-action value')
#                 plt.show()
#
            break
#
        RL.store_transition(observation, action, reward)
        observation = observation_
        trj.append(observation_)
        #print("length of memory :", len(RL.ep_rs))
    trjs.append(trj)
    if i_episode%10==0:
        trjs = generate_trjs(M_trjs, N_steps, RL, env)
        P = RL.stochastic_trjs(trjs)
    #print("P \n", P)
        Im = RL.Im_s(P, tempn)
        print("Im \n", Im)
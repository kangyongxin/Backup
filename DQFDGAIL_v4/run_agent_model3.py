# -*- coding: utf-8 -*
import tensorflow as tf
import gym
from gym import wrappers
from my_Config3 import Config
import random
import numpy as np
import pickle
from collections import deque
import itertools
from my_Memory3 import Memory
from _AGENT3 import AGENT3

def set_n_step(container, n):
    t_list = list(container)
    # accumulated reward of first (trajectory_n-1) transitions
    n_step_reward = sum([t[2] * Config.GAMMA**i for i, t in enumerate(t_list[0:min(len(t_list), n) - 1])])
    for begin in range(len(t_list)):
        end = min(len(t_list) - 1, begin + Config.trajectory_n - 1)
        n_step_reward += t_list[end][2]*Config.GAMMA**(end-begin)
        # extend[n_reward, n_next_s, n_done, actual_n]
        t_list[begin].extend([n_step_reward, t_list[end][3], t_list[end][4], end-begin+1])
        n_step_reward = (n_step_reward - t_list[begin][2])/Config.GAMMA
    return t_list


if __name__ == '__main__':
    env = gym.make(Config.ENV_NAME)
    # load demo to memory
    with open(Config.DEMO_DATA_PATH, 'rb') as f:
        demo_transitions = pickle.load(f)
        demo_transitions = deque(itertools.islice(demo_transitions, 0, Config.DEMO_BUFFER_SIZE))
        print("demo_transitions len: ", len(demo_transitions))
    agent = AGENT3(env, Config)
    agent.add_data_to_memory(demo_transitions, agent.demo_memory)
    # print("demo_memory", agent.get_data_from_fullmemory(agent.demo_memory))
    while not agent.replay_memory.full():
        agent.copy_AFULL_to_B(agent.demo_memory, agent.replay_memory)
    # print("replay_memory", agent.get_data_from_fullmemory(agent.replay_memory))
    agent.restore_model()
    agent.epsilon=0.01
    iteration = int(100000)
    action_dim = env.action_space.n
    while True:
        done = False
        obs = env.reset()
        score = 0
        while not done:
            act = agent.egreedy_action(obs)
            next_obs, reward, done, info = env.step(act)
            score += reward
            env.render(next_obs)
            obs = next_obs
        if done:
            print("score:{}".format(score))
    #env.close()
    print("done")
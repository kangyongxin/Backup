# -*- coding: utf-8 -*
import tensorflow as tf
import random
import numpy as np
import pickle
from collections import deque
import itertools
from FMemory import Memory
from FAGENT import AGENT

def set_n_step(container, n, Config):
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

def run_DQfD(env, Config):
    sess=tf.InteractiveSession()
    with open(Config.DEMO_DATA_PATH, 'rb') as f:
        demo_transitions = pickle.load(f)
        demo_transitions = deque(itertools.islice(demo_transitions, 0, Config.DEMO_BUFFER_SIZE))
        print("demo_transitions len: ", len(demo_transitions))
    with tf.variable_scope('AGENT'):
        agent = AGENT(env, Config,sess)
    agent.add_data_to_memory(demo_transitions, agent.demo_memory)
    #print("demo_memory", agent.get_data_from_fullmemory(agent.demo_memory))
    agent.copy_AFULL_to_B(agent.demo_memory, agent.replay_memory)
    try:
        print("agent model existed")
        agent.restore_model()
        agent_model_improve_flag = False
        agent.epsilon = 0.01
    except:
        print("there is no model,we are going to initialize it randomly")
        agent.sess.run(tf.global_variables_initializer())
        print("agent.epsilon:{}".format(agent.epsilon))
        agent.save_model()
        agent_model_improve_flag = True
    scores, e, replay_full_episode = [], 0, None
    n_dqfd = 0
    while True:
        if agent_model_improve_flag:
            agent.restore_model()
            agent_model_improve_flag = False
        e += 1
        done, score, n_step_reward, state = False, 0, None, env.reset()
        t_q = deque(maxlen=Config.trajectory_n)
        n_dqfd += 1
        while done is False:
            action = agent.egreedy_action(state)
            next_state, reward, done, _ = env.step(action)
            score += reward
            reward = reward if not done or score == 499 else -100
            reward_to_sub = 0. if len(t_q) < t_q.maxlen else t_q[0][2]  # record the earliest reward for the sub

            t_q.append([state, action, reward, next_state, done, 0.0])
            if len(t_q) == t_q.maxlen:
                if n_step_reward is None:  # only compute once when t_q first filled
                    n_step_reward = sum([t[2] * Config.GAMMA ** i for i, t in enumerate(t_q)])
                else:
                    n_step_reward = (n_step_reward - reward_to_sub) / Config.GAMMA
                    n_step_reward += reward * Config.GAMMA ** (Config.trajectory_n - 1)
                t_q[0].extend([n_step_reward, next_state, done, t_q.maxlen])  # actual_n is max_len here
                agent.perceive(t_q[0])  # perceive when a transition is completed
                if agent.replay_memory.full():
                    agent.train_Q_network(update=False)  # train along with generation
                    replay_full_episode = replay_full_episode or e
            state = next_state
            env.render(state)

            if done:
                # handle transitions left in t_q
                t_q.popleft()  # first transition's n-step is already set
                transitions = set_n_step(t_q, Config.trajectory_n, Config)
                for t in transitions:
                    agent.perceive(t)
                    if agent.replay_memory.full():
                        agent.train_Q_network(update=False)
                        replay_full_episode = replay_full_episode or e
                if agent.replay_memory.full():
                    scores.append(score)
                    agent.sess.run(agent.update_target_net)
                if replay_full_episode is not None:
                    print("episode: {}  trained-episode: {}  score: {}  memory length: {}  epsilon: {}"
                          .format(e, e - replay_full_episode, score, len(agent.replay_memory), agent.epsilon))
                if agent.epsilon == agent.config.FINAL_EPSILON:
                    agent.save_model()
                    agent_model_improve_flag = True

        if len(scores) > 100:
            break
    return scores



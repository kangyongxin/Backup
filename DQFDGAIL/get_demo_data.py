# to get demo data by DDQN
import tensorflow as tf
import numpy as np
import pickle
from my_DQfDDDQN import DQfDDDQN
from my_Config import Config, DDQNConfig, DQfDConfig
from collections import deque
import itertools
import gym

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

def get_demo_data(env):
    # env = wrappers.Monitor(env, '/tmp/CartPole-v0', force=True)
    # agent.restore_model()
    with tf.variable_scope('get_demo_data'):
        agent = DQfDDDQN(env, DDQNConfig())

    e = 0
    while True:
        done = False
        score = 0  # sum of reward in one episodeset_n_step
        state = env.reset()
        demo = []
        while done is False:
            action = agent.egreedy_action(state)  # e-greedy action for train
            next_state, reward, done, _ = env.step(action)
            #print("reward", reward)
            score += reward
            reward = reward if not done or score == 499 else -100
            #print("score",score)
            agent.perceive([state, action, reward, next_state, done, 0.0])  # 0. means it is not a demo data
            demo.append([state, action, reward, next_state, done, 1.0])  # record the data that could be expert-data
            agent.train_Q_network(update=True)#False)
            state = next_state
            env.render(state)#20190121
        if done:
            if score >= -199:  # expert demo data
                demo = set_n_step(demo, Config.trajectory_n)
                print(demo)
                agent.demo_buffer.extend(demo)
                with open(Config.DEMO_DATA_PATH, 'wb') as f:
                    pickle.dump(agent.demo_buffer, f, protocol=2)
            agent.sess.run(agent.update_target_net)
            print("episode:", e, "  score:", score, "  demo_buffer:", len(agent.demo_buffer),
                  "  memory length:", len(agent.replay_buffer), "  epsilon:", agent.epsilon)
            if len(agent.demo_buffer) >= Config.demo_buffer_size:
                agent.demo_buffer = deque(itertools.islice(agent.demo_buffer, 0, Config.demo_buffer_size))
                break
        e += 1



if __name__ == '__main__':

    env = gym.make(Config.ENV_NAME)
    # env = wrappers.Monitor(env, '/tmp/CartPole-v0', force=True)
    # ------------------------ get demo scores by DDQN -----------------------------
    get_demo_data(env)#20190121
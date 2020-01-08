# -*- coding: utf-8 -*
import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import tensorflow as tf
from gym import wrappers
import gym
import numpy as np
import pickle
from my_Config import Config, DDQNConfig, DQfDConfig
from my_DQfD import DQfD
#from DQfDDDQN import DQfDDDQN
from collections import deque
import itertools
#
from ppo import PPOTrain
from discriminator import Discriminator
from policy_net import Policy_net
from my_GAIL import GAIL


def map_scores(dqfd_scores=None, ddqn_scores=None, xlabel=None, ylabel=None):
    if dqfd_scores is not None:
        plt.plot(dqfd_scores, 'r')
    if ddqn_scores is not None:
        plt.plot(ddqn_scores, 'b')
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    plt.show()

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

def run(index,env):
    with open(Config.DEMO_DATA_PATH, 'rb') as f:
        demo_transitions = pickle.load(f)
        demo_transitions = deque(itertools.islice(demo_transitions, 0, Config.demo_buffer_size))
        assert len(demo_transitions) == Config.demo_buffer_size

    with tf.variable_scope('DQfD_' + str(index)):
        agent = DQfD(env, DQfDConfig(), demo_transitions=demo_transitions)


    index_gail=index
    with tf.variable_scope('gail' + str(index_gail)):
        soul = GAIL(DQfDConfig, env, demo_transitions)

    agent.pre_train()  # use the demo data to pre-train network
    scores, e, replay_full_episode = [], 0, None
    n_dqfd = 0
    while True:
        n_dqfd = n_dqfd + 1
        done, score, n_step_reward, state = False, 0, None, env.reset()
        t_q = deque(maxlen=Config.trajectory_n)
        while done is False:
            action = agent.egreedy_action(state)  # e-greedy action for train
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
                transitions = set_n_step(t_q, Config.trajectory_n)
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
                          .format(e, e-replay_full_episode, score, len(agent.replay_memory), agent.epsilon))
                # if np.mean(scores[-min(10, len(scores)):]) > 495:
                #     break
                env.close()
                print("n_dqfd", n_dqfd, "dqfd scores :", score)
                if n_dqfd > 50:
                    n_dqfd = 0
                    soul.expert_memory = agent.demo_memory
                    _, demo_transitions, _ = soul.expert_memory.sample(soul.config.BATCH_SIZE)
                    expert_observations = [data[0] for data in demo_transitions]
                    expert_actions = [data[1] for data in demo_transitions]
                    obs = env.reset()
                    success_num = 0
                    iteration = int(2000)  # 0319
                    for iteration in range(iteration):
                        # print("running policy ")
                        observations = []
                        # states_for_memory=[]
                        actions = []
                        # do NOT use rewards to update policy , # 0319 why ?
                        rewards = []
                        v_preds = []
                        run_policy_steps = 0
                        score = 0
                        t_q = deque(maxlen=Config.trajectory_n)
                        done, score, n_step_reward, state_for_memory = False, 0, None, env.reset()
                        while True:
                            run_policy_steps += 1
                            obs = np.stack([obs]).astype(dtype=np.float32)  # prepare to feed placeholder Policy.obs
                            act, v_pred = soul.Policy.act(obs=obs, stochastic=True)

                            act = np.asscalar(act)
                            v_pred = np.asscalar(v_pred)
                            next_obs, reward, done, info = env.step(act)
                            next_state_for_memory = next_obs
                            score += reward
                            reward_to_sub = 0. if len(t_q) < t_q.maxlen else t_q[0][
                                2]  # record the earliest reward for the sub
                            t_q.append([state_for_memory, act, reward, next_state_for_memory, done, 0.0])
                            if len(t_q) == t_q.maxlen:
                                if n_step_reward is None:  # only compute once when t_q first filled
                                    n_step_reward = sum([t[2] * Config.GAMMA ** i for i, t in enumerate(t_q)])
                                else:
                                    n_step_reward = (n_step_reward - reward_to_sub) / Config.GAMMA
                                    n_step_reward += reward * Config.GAMMA ** (Config.trajectory_n - 1)
                                t_q[0].extend([n_step_reward, next_state_for_memory, done,
                                               t_q.maxlen])  # actual_n is max_len here
                                # agent.perceive(t_q[0])  # perceive when a transition is completed

                            env.render()  # 0313
                            observations.append(obs)
                            actions.append(act)
                            rewards.append(reward)
                            v_preds.append(v_pred)

                            if done:
                                t_q.popleft()  # first transition's n-step is already set
                                transitions = set_n_step(t_q, Config.trajectory_n)
                                next_obs = np.stack([next_obs]).astype(
                                    dtype=np.float32)  # prepare to feed placeholder Policy.obs
                                _, v_pred = soul.Policy.act(obs=next_obs, stochastic=True)
                                v_preds_next = v_preds[1:] + [np.asscalar(v_pred)]
                                obs = env.reset()
                                print("iteration", iteration, "score", score)
                                break
                            else:
                                obs = next_obs
                                state_for_memory = next_state_for_memory
                        #
                        if sum(rewards) >= 100:
                            soul.save_model()
                            success_num += 1

                            # todo
                            # 在 能够得到较好的回报 的时候 存储这次的demo
                            for t in transitions:
                                soul.perceive(t)
                                soul.generate_memory.memory_len()
                            if success_num >= 10:
                                print('Clear!! Model saved.')
                                env.close()
                                break
                        else:
                            success_num = 0

                        # convert list to numpy array for feeding tf.placeholder
                        observations = np.reshape(observations, newshape=[-1] + list(soul.ob_space.shape))
                        actions = np.array(actions).astype(dtype=np.int32)

                        # train discriminator
                        for i in range(2):
                            # print("training D")
                            soul.D.train(expert_s=expert_observations,
                                         expert_a=expert_actions,
                                         agent_s=observations,
                                         agent_a=actions)

                        # output of this discriminator is reward
                        d_rewards = soul.D.get_rewards(agent_s=observations, agent_a=actions)
                        d_rewards = np.reshape(d_rewards, newshape=[-1]).astype(dtype=np.float32)

                        gaes = soul.PPO.get_gaes(rewards=d_rewards, v_preds=v_preds, v_preds_next=v_preds_next)
                        gaes = np.array(gaes).astype(dtype=np.float32)
                        # gaes = (gaes - gaes.mean()) / gaes.std()
                        v_preds_next = np.array(v_preds_next).astype(dtype=np.float32)

                        # train policy
                        inp = [observations, actions, gaes, d_rewards, v_preds_next]
                        soul.PPO.assign_policy_parameters()
                        for epoch in range(6):
                            # print("updating PPO ")
                            sample_indices = np.random.randint(low=0, high=observations.shape[0],
                                                               size=32)  # indices are in [low, high)
                            sampled_inp = [np.take(a=a, indices=sample_indices, axis=0) for a in
                                           inp]  # sample training data
                            soul.PPO.train(obs=sampled_inp[0],
                                           actions=sampled_inp[1],
                                           gaes=sampled_inp[2],
                                           rewards=sampled_inp[3],
                                           v_preds_next=sampled_inp[4])

                    if soul.generate_memory.full():
                        _, data, _ = soul.generate_memory.sample(soul.generate_memory.memory_len())
                        for t in data:
                            agent.perceive(t)
                            agent.replay_memory.memory_len()

if __name__ == '__main__':

    env = gym.make(Config.ENV_NAME)

    for i in range(3):
        run(1, env)






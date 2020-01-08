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
import matplotlib
import matplotlib.pyplot as plt
from SOUL3 import SOUL3
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
def run_soul(env, soul):
    #todo using the expert data to generate memory
    print("am i right")
    expert_data = soul.get_data_from_fullmemory(soul.expert_memory)
    expert_observations = [data[0] for data in expert_data]
    expert_actions = [data[1] for data in expert_data]


    try:
        print("soul model existed")
        soul.restore_model()
        model_improve_flag = False
        # agent.epsilon = 0.01
    except:
        print("there is no model,we are going to initialize it randomly")
        soul.Soulsess.run(tf.global_variables_initializer())
        #print("soul.epsilon:{}".format(soul.epsilon))
        soul.save_model()
        model_improve_flag = True
    obs = env.reset()
    success_num = 0
    for i in range(100000):
        if model_improve_flag:
            soul.restore_model()
            model_improve_flag=False
        observations = []
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
            score += reward

            next_state_for_memory = next_obs
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
                print("iteration", i, "score", score)
                break
            else:
                obs = next_obs
                state_for_memory = next_state_for_memory
        if sum(rewards) >= 499:
            soul.save_model()
            model_improve_flag =True
            success_num += 1

            # todo
            # 在 能够得到较好的回报 的时候 存储这次的demo
            for t in transitions:
                soul.perceive(t)
                soul.generate_memory.memory_len()
            if soul.generate_memory.full() and success_num>20:
                print('Clear!! Model saved.')
                env.close()
                break
        # else:
        #     success_num = 0

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


if __name__ == '__main__':
    env = gym.make(Config.ENV_NAME)
    soul_flag = True
    #print("env.observation_space:{}",env.observation_space)

    #load demo to memory
    with open(Config.DEMO_DATA_PATH, 'rb') as f:
        demo_transitions = pickle.load(f)
        demo_transitions = deque(itertools.islice(demo_transitions, 0, Config.DEMO_BUFFER_SIZE))
        print("demo_transitions len: ", len(demo_transitions))
    agent = AGENT3(env, Config)
    agent.add_data_to_memory( demo_transitions, agent.demo_memory)
    #print("demo_memory", agent.get_data_from_fullmemory(agent.demo_memory))
    # while not agent.replay_memory.full():
    agent.copy_AFULL_to_B(agent.demo_memory, agent.replay_memory)
    # print("replay_memory", agent.get_data_from_fullmemory(agent.replay_memory))
    try:
        print("agent model existed")
        agent.restore_model()
        agent_model_improve_flag = False
        agent.epsilon = 0.01
    except:
        print("there is no model,we are going to initialize it randomly")
        agent.Agentsess.run(tf.global_variables_initializer())
        print("agent.epsilon:{}".format(agent.epsilon))
        agent.save_model()
        agent_model_improve_flag = True

    agent.pre_train()  # use the demo data to pre-train network
    agent.save_model()
    agent_model_improve_flag = True
    if soul_flag:
        soul = SOUL3(env, Config)
        soul.add_data_to_memory(demo_transitions, soul.expert_memory)
        #print("demo_memory", soul.get_data_from_fullmemory(soul.expert_memory))
        run_soul(env, soul)

        data = soul.get_data_from_fullmemory(soul.generate_memory)
        agent.add_data_to_memory(data, agent.replay_memory)

    #agent.restore_model()
    scores, e, replay_full_episode = [], 0, None
    n_dqfd = 0


    while True:
        if agent_model_improve_flag:
            agent.restore_model()
            agent_model_improve_flag=False
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
                transitions = set_n_step(t_q, Config.trajectory_n)
                for t in transitions:
                    agent.perceive(t)
                    if agent.replay_memory.full():
                        agent.train_Q_network(update=False)
                        replay_full_episode = replay_full_episode or e
                if agent.replay_memory.full():
                    scores.append(score)
                    agent.Agentsess.run(agent.update_target_net)
                if replay_full_episode is not None:
                    print("episode: {}  trained-episode: {}  score: {}  memory length: {}  epsilon: {}"
                          .format(e, e-replay_full_episode, score, len(agent.replay_memory), agent.epsilon))
                if agent.epsilon == agent.config.FINAL_EPSILON:
                    agent.save_model()
                    agent_model_improve_flag = True
                # if np.mean(scores[-min(10, len(scores)):]) > 495:
                #     break
                #env.close()
        if soul_flag:
            if n_dqfd >100:
                print("need to improve the ppo")
                run_soul(env,soul)
                n_dqfd = 0
        if len(scores)>100:
            break
    plt.plot(scores, 'r')
    plt.show()
    # iteration = int(1000)
    # action_dim = env.action_space.n
    # for i in range(iteration):
    #     done = False
    #     obs = env.reset()
    #     while not done:
    #         act = agent.random_action()
    #         next_obs, reward, done, info = env.step(act)
    #         env.render(next_obs)
    #         obs = next_obs
    # env.close()
    # print("done")




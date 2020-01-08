import tensorflow as tf
import random
import numpy as np
import pickle
from collections import deque
import itertools
from FMemory import Memory
from FAGENT import AGENT
from FSOUL import SOUL


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

def run_GAIL(env, Config):
    sess = tf.InteractiveSession()
    with open(Config.DEMO_DATA_PATH, 'rb') as f:
        demo_transitions = pickle.load(f)
        demo_transitions = deque(itertools.islice(demo_transitions, 0, Config.DEMO_BUFFER_SIZE))
        print("demo_transitions len: ", len(demo_transitions))
    with tf.variable_scope('SOUL'):
        soul = SOUL(env, Config, sess)
    soul.add_data_to_memory(demo_transitions, soul.expert_memory)
    try:
        print("soul model existed")
        soul.restore_model()
        soul_model_improve_flag = False
        # agent.epsilon = 0.01
    except:
        print("there is no model,we are going to initialize it randomly")
        soul.sess.run(tf.global_variables_initializer())
        # print("soul.epsilon:{}".format(soul.epsilon))
        soul.save_model()
        soul_model_improve_flag = True

    expert_data = soul.get_data_from_fullmemory(soul.expert_memory)
    expert_observations = [data[0] for data in expert_data]
    expert_actions = [data[1] for data in expert_data]

    obs = env.reset()
    success_num = 0
    scores = []
    for i in range(100000):
        if soul_model_improve_flag:
            soul.restore_model()
            soul_model_improve_flag = False
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
                transitions = set_n_step(t_q, Config.trajectory_n,Config)

                next_obs = np.stack([next_obs]).astype(
                    dtype=np.float32)  # prepare to feed placeholder Policy.obs
                _, v_pred = soul.Policy.act(obs=next_obs, stochastic=True)
                v_preds_next = v_preds[1:] + [np.asscalar(v_pred)]
                obs = env.reset()
                print("iteration", i, "score", score)
                scores.append(score)
                break
            else:
                obs = next_obs
                state_for_memory = next_state_for_memory
        if sum(rewards) >= 499:
            soul.save_model()
            model_improve_flag = True
            success_num += 1

            # todo
            # 在 能够得到较好的回报 的时候 存储这次的demo
            for t in transitions:
                soul.perceive(t)
                soul.generate_memory.memory_len()
            if soul.generate_memory.full() and success_num > 100:
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
    return scores
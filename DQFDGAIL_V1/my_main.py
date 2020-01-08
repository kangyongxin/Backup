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
def run_gail1(soul,env,index_gail):
    _, demo_transitions, _ = soul.expert_memory.sample(soul.config.BATCH_SIZE)
    expert_observations = [data[0] for data in demo_transitions]
    expert_actions = [data[1] for data in demo_transitions]


    with tf.Session() as sess:
        obs = env.reset()
        if index_gail >1:
            soul.restore_model()
        sess.run(tf.global_variables_initializer())
        state_for_memory=obs#为了处理两套程序中使用的数据格式不同
        success_num = 0
        iteration = int(1000) #0319
        for iteration in range(iteration):
            #print("running policy ")
            observations = []
            #states_for_memory=[]
            actions = []
            # do NOT use rewards to update policy , # 0319 why ?
            rewards = []
            v_preds = []
            run_policy_steps = 0
            score=0
            t_q = deque(maxlen=Config.trajectory_n)
            done, score, n_step_reward, state_for_memory = False, 0, None, env.reset()
            while True:
                run_policy_steps += 1
                obs = np.stack([obs]).astype(dtype=np.float32)  # prepare to feed placeholder Policy.obs
                act, v_pred = soul.Policy.act(obs=obs, stochastic=True)

                act = np.asscalar(act)
                v_pred = np.asscalar(v_pred)
                next_obs, reward, done, info = env.step(act)
                next_state_for_memory=next_obs
                score += reward
                reward_to_sub = 0. if len(t_q) < t_q.maxlen else t_q[0][2]  # record the earliest reward for the sub
                t_q.append([state_for_memory, act, reward, next_state_for_memory, done, 0.0])
                if len(t_q) == t_q.maxlen:
                    if n_step_reward is None:  # only compute once when t_q first filled
                        n_step_reward = sum([t[2] * Config.GAMMA ** i for i, t in enumerate(t_q)])
                    else:
                        n_step_reward = (n_step_reward - reward_to_sub) / Config.GAMMA
                        n_step_reward += reward * Config.GAMMA ** (Config.trajectory_n - 1)
                    t_q[0].extend([n_step_reward, next_state_for_memory, done, t_q.maxlen])  # actual_n is max_len here
                    #agent.perceive(t_q[0])  # perceive when a transition is completed

                env.render()  # 0313
                observations.append(obs)
                actions.append(act)
                rewards.append(reward)
                v_preds.append(v_pred)

                if done:
                    t_q.popleft()  # first transition's n-step is already set
                    transitions = set_n_step(t_q, Config.trajectory_n)
                    next_obs = np.stack([next_obs]).astype(dtype=np.float32)  # prepare to feed placeholder Policy.obs
                    _, v_pred = soul.Policy.act(obs=next_obs, stochastic=True)
                    v_preds_next = v_preds[1:] + [np.asscalar(v_pred)]
                    obs = env.reset()
                    print("iteration",iteration,"score", score)
                    break
                else:
                    obs = next_obs
                    state_for_memory=next_state_for_memory
                #print("state_for memory",state_for_memory)
            #writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='episode_length', simple_value=run_policy_steps)]), iteration)
            #writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='episode_reward', simple_value=sum(rewards))]), iteration)

            #
            if sum(rewards) >= 100:
                soul.save_model()
                success_num += 1

                # todo
                # 在 能够得到较好的回报 的时候 存储这次的demo
                for t in transitions:
                    soul.perceive(t)
                    soul.generate_memory.memory_len()

                if success_num >= 3:

                    #saver.save(sess, 'trained_models/model.ckpt')
                    #saver.save(sess, 'trained_models/model' + str(index_gail) + '.ckpt')
                    #print(success_num)
                    print('Clear!! Model saved.')
                    env.close()
                    break
            else:
                success_num = 0

            # convert list to numpy array for feeding tf.placeholder
            observations = np.reshape(observations, newshape=[-1] + list(soul.ob_space.shape))
            actions = np.array(actions).astype(dtype=np.int32)

            # train discriminator
            for i in range(20):
                #print("training D")
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
            for epoch in range(2):
                #print("updating PPO ")
                sample_indices = np.random.randint(low=0, high=observations.shape[0],
                                                   size=32)  # indices are in [low, high)
                sampled_inp = [np.take(a=a, indices=sample_indices, axis=0) for a in inp]  # sample training data
                soul.PPO.train(obs=sampled_inp[0],
                          actions=sampled_inp[1],
                          gaes=sampled_inp[2],
                          rewards=sampled_inp[3],
                          v_preds_next=sampled_inp[4])

def run_gail(agent,index_gail,env):
    DG_flag=1
    #env.seed(0)
    ob_space = env.observation_space
    Policy = Policy_net('policy_'+str(index_gail), env)
    Old_Policy = Policy_net('old_policy'+str(index_gail), env)
    gamma=0.95
    PPO = PPOTrain(Policy, Old_Policy, gamma)
    D = Discriminator(env,index_gail)

    if DG_flag:
        # with open(Config.DEMO_DATA_PATH, 'rb') as f:
        #     demo_transitions = pickle.load(f)
        #     demo_transitions = deque(itertools.islice(demo_transitions, 0, Config.demo_buffer_size))
        #     assert len(demo_transitions) == Config.demo_buffer_size
        expert_data = agent.replay_memory if agent.replay_memory.full() else agent.demo_memory
        _, demo_transitions, _ = expert_data.sample(agent.config.BATCH_SIZE)
        expert_observations = [data[0] for data in demo_transitions]
        expert_actions = [data[1] for data in demo_transitions]
    else :
        expert_observations = np.genfromtxt('trajectory/observations.csv')
        expert_actions = np.genfromtxt('trajectory/actions.csv', dtype=np.int32)



    with tf.Session() as sess:
        # writer = tf.summary.FileWriter(args.logdir, sess.graph)
        #load_path=saver.restore(sess,"trained_models/model.ckpt")
        #sess.run(tf.global_variables_initializer())
        #if index_gail>1:
        #   saver.restore(sess, 'trained_models/model' + str(index_gail-1) + '.ckpt')

        obs = env.reset()
        state_for_memory=obs#为了处理两套程序中使用的数据格式不同
        success_num = 0
        iteration = int(2000) #0319
        for iteration in range(iteration):
            #print("running policy ")
            observations = []
            #states_for_memory=[]
            actions = []
            # do NOT use rewards to update policy , # 0319 why ?
            rewards = []
            v_preds = []
            run_policy_steps = 0
            score=0
            if DG_flag:
                t_q = deque(maxlen=Config.trajectory_n)
                done, score, n_step_reward, state_for_memory = False, 0, None, env.reset()
            while True:
                run_policy_steps += 1
                obs = np.stack([obs]).astype(dtype=np.float32)  # prepare to feed placeholder Policy.obs
                act, v_pred = Policy.act(obs=obs, stochastic=True)

                act = np.asscalar(act)
                v_pred = np.asscalar(v_pred)
                next_obs, reward, done, info = env.step(act)
                next_state_for_memory=next_obs
                score += reward
                if DG_flag:
                    reward_to_sub = 0. if len(t_q) < t_q.maxlen else t_q[0][2]  # record the earliest reward for the sub
                    t_q.append([state_for_memory, act, reward, next_state_for_memory, done, 0.0])
                    if len(t_q) == t_q.maxlen:
                        if n_step_reward is None:  # only compute once when t_q first filled
                            n_step_reward = sum([t[2] * Config.GAMMA ** i for i, t in enumerate(t_q)])
                        else:
                            n_step_reward = (n_step_reward - reward_to_sub) / Config.GAMMA
                            n_step_reward += reward * Config.GAMMA ** (Config.trajectory_n - 1)
                        t_q[0].extend([n_step_reward, next_state_for_memory, done, t_q.maxlen])  # actual_n is max_len here
                        #agent.perceive(t_q[0])  # perceive when a transition is completed

                env.render()  # 0313
                observations.append(obs)
                actions.append(act)
                rewards.append(reward)
                v_preds.append(v_pred)

                if done:
                    if DG_flag:
                        t_q.popleft()  # first transition's n-step is already set
                        transitions = set_n_step(t_q, Config.trajectory_n)
                    next_obs = np.stack([next_obs]).astype(dtype=np.float32)  # prepare to feed placeholder Policy.obs
                    _, v_pred = Policy.act(obs=next_obs, stochastic=True)
                    v_preds_next = v_preds[1:] + [np.asscalar(v_pred)]
                    obs = env.reset()
                    print("iteration",iteration,"score", score)
                    break
                else:
                    obs = next_obs
                    state_for_memory=next_state_for_memory
                #print("state_for memory",state_for_memory)
            #writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='episode_length', simple_value=run_policy_steps)]), iteration)
            #writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='episode_reward', simple_value=sum(rewards))]), iteration)

            #
            if sum(rewards) >= 100:

                success_num += 1
                # todo
                # 在 能够得到较好的回报 的时候 存储这次的demo
                if DG_flag:
                    for t in transitions:
                        agent.perceive(t)
                        agent.replay_memory.memory_len()

                if success_num >= 3:
                    #saver.save(sess, 'trained_models/model.ckpt')
                    #saver.save(sess, 'trained_models/model' + str(index_gail) + '.ckpt')
                    print(success_num)
                    print('Clear!! Model saved.')
                    env.close()
                    break
            else:
                success_num = 0

            # convert list to numpy array for feeding tf.placeholder
            observations = np.reshape(observations, newshape=[-1] + list(ob_space.shape))
            actions = np.array(actions).astype(dtype=np.int32)

            # train discriminator
            for i in range(2):
                #print("training D")
                D.train(expert_s=expert_observations,
                        expert_a=expert_actions,
                        agent_s=observations,
                        agent_a=actions)

            # output of this discriminator is reward
            d_rewards = D.get_rewards(agent_s=observations, agent_a=actions)
            d_rewards = np.reshape(d_rewards, newshape=[-1]).astype(dtype=np.float32)

            gaes = PPO.get_gaes(rewards=d_rewards, v_preds=v_preds, v_preds_next=v_preds_next)
            gaes = np.array(gaes).astype(dtype=np.float32)
            # gaes = (gaes - gaes.mean()) / gaes.std()
            v_preds_next = np.array(v_preds_next).astype(dtype=np.float32)

            # train policy
            inp = [observations, actions, gaes, d_rewards, v_preds_next]
            PPO.assign_policy_parameters()
            for epoch in range(6):
                #print("updating PPO ")
                sample_indices = np.random.randint(low=0, high=observations.shape[0],
                                                   size=32)  # indices are in [low, high)
                sampled_inp = [np.take(a=a, indices=sample_indices, axis=0) for a in inp]  # sample training data
                PPO.train(obs=sampled_inp[0],
                          actions=sampled_inp[1],
                          gaes=sampled_inp[2],
                          rewards=sampled_inp[3],
                          v_preds_next=sampled_inp[4])

            # summary = PPO.get_summary(obs=inp[0],
            #                           actions=inp[1],
            #                           gaes=inp[2],
            #                           rewards=inp[3],
            #                           v_preds_next=inp[4])


            #writer.add_summary(summary, iteration)
        #writer.close()

#
# def run_DDQN(index, env):
#     with tf.variable_scope('DDQN_' + str(index)):
#         agent = DQfDDDQN(env, DDQNConfig())
#     scores = []
#     for e in range(Config.episode):
#         done = False
#         score = 0  # sum of reward in one episode
#         state = env.reset()
#         while done is False:
#             action = agent.egreedy_action(state)  # e-greedy action for train
#             next_state, reward, done, _ = env.step(action)
#             score += reward
#             reward = reward if not done or score == 499 else -100
#             agent.perceive([state, action, reward, next_state, done, 0.0])  # 0. means it is not a demo data
#             agent.train_Q_network(update=False)
#             state = next_state
#             env.render(state)
#
#         if done:
#             scores.append(score)
#             agent.sess.run(agent.update_target_net)
#             print("episode:", e, "  score:", score, "  demo_buffer:", len(agent.demo_buffer),
#                   "  memory length:", len(agent.replay_buffer), "  epsilon:", agent.epsilon)
#             # if np.mean(scores[-min(10, len(scores)):]) > 490:
#             #     break
#     return scores


def run_DQfD(index, env):
    with open(Config.DEMO_DATA_PATH, 'rb') as f:
        demo_transitions = pickle.load(f)
        demo_transitions = deque(itertools.islice(demo_transitions, 0, Config.demo_buffer_size))
        assert len(demo_transitions) == Config.demo_buffer_size

    with tf.variable_scope('DQfD_' + str(index)):
        agent = DQfD(env, DQfDConfig(), demo_transitions=demo_transitions)

    #print("demo_transiions", demo_transitions[1])
    agent.pre_train()  # use the demo data to pre-train network
    index_gail=0
    scores, e, replay_full_episode = [], 0, None
    n_dqfd=0
    while True:
        n_dqfd=n_dqfd+1

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
                    n_step_reward = sum([t[2]*Config.GAMMA**i for i, t in enumerate(t_q)])
                else:
                    n_step_reward = (n_step_reward - reward_to_sub) / Config.GAMMA
                    n_step_reward += reward*Config.GAMMA**(Config.trajectory_n-1)
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
            #agent.sess.close()
            #agent.save_model()
            if n_dqfd >20:
                n_dqfd=0
                break
        if len(scores) >= Config.episode:
            break
        e += 1

    # index_gail=index
    # print("index_gail ________________________________", index_gail)
    # with tf.variable_scope('gail' + str(index_gail)):
    #     soul = GAIL(DQfDConfig, env, demo_transitions)
    # soul.expert_memory = agent.demo_memory
    #
    # run_gail1(soul, env, index_gail)
    # if soul.generate_memory.full():
    #     _, data, _ = soul.generate_memory.sample(soul.generate_memory.memory_len())
    #
    #     for t in data:
    #         agent.perceive(t)
    #         agent.replay_memory.memory_len()
    #
    return scores


# extend [n_step_reward, n_step_away_state] for transitions in demo
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
            agent.sess.run(agent.update_target_net)
            print("episode:", e, "  score:", score, "  demo_buffer:", len(agent.demo_buffer),
                  "  memory length:", len(agent.replay_buffer), "  epsilon:", agent.epsilon)
            if len(agent.demo_buffer) >= Config.demo_buffer_size:
                agent.demo_buffer = deque(itertools.islice(agent.demo_buffer, 0, Config.demo_buffer_size))
                break
        e += 1

    with open(Config.DEMO_DATA_PATH, 'wb') as f:
        pickle.dump(agent.demo_buffer, f, protocol=2)


if __name__ == '__main__':

    env = gym.make(Config.ENV_NAME)
    # env = wrappers.Monitor(env, '/tmp/CartPole-v0', force=True)
    # ------------------------ get demo scores by DDQN -----------------------------
    # get_demo_data(env)#20190121
    # --------------------------  get DDQN scores ----------------------------------
    # ddqn_sum_scores = np.zeros(Config.episode)
    # for i in range(Config.iteration):
    #
    #    scores = run_DDQN(i, env)
    #    ddqn_sum_scores = np.array([a + b for a, b in zip(scores, ddqn_sum_scores)])
    # ddqn_mean_scores = ddqn_sum_scores / Config.iteration
    # with open('./ddqn_mean_scores.p', 'wb') as f:
    #    pickle.dump(ddqn_mean_scores, f, protocol=2)
    # with open('./ddqn_mean_scores.p', 'rb') as f:
    #    ddqn_mean_scores = pickle.load(f)
    # ----------------------------- get DQfD scores --------------------------------
    # dqfd_sum_scores = np.zeros(Config.episode)
    # for i in range(Config.iteration):
    #     scores = run_DQfD(i, env)
    #     dqfd_sum_scores = np.array([a + b for a, b in zip(scores, dqfd_sum_scores)])
    # dqfd_mean_scores = dqfd_sum_scores / Config.iteration
    # with open('./dqfd_mean_scores.p', 'wb') as f:
    #    pickle.dump(dqfd_mean_scores, f, protocol=2)
    #
    # map_scores(dqfd_scores=dqfd_mean_scores, ddqn_scores=ddqn_mean_scores,
    #     xlabel='Red: dqfd         Blue: ddqn', ylabel='Scores')
    # env.close()
    # # gym.upload('/tmp/carpole_DDQN-1', api_key='sk_VcAt0Hh4RBiG2yRePmeaLA')
    #-------------------------------run_gail-------------------------------------
    for i in range(3):
        run_DQfD(1,env)





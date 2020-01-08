# -*- coding: utf-8 -*-
import os


class Config:
    ENV_NAME ="MountainCar-v0"#"Seaquest-v0"#BipedalWalker-v2"#LunarLander-v2" #Acrobot-v1"#"MountainCar-v0"#"CartPole-v1"# "CartPole-v0"#"MountainCar-v0"#"#"Acrobot-v1"#
    GAMMA = 0.99  # discount factor for target Q
    INITIAL_EPSILON = 1.0  # starting value of epsilon
    FINAL_EPSILON = 0.01 # final value of epsilon
    EPSILIN_DECAY = 0.999
    START_TRAINING = 1000  # experience replay buffer size
    BATCH_SIZE = 64  # size of minibatch
    UPDATE_TARGET_NET = 20 # update eval_network params every 200 steps
    LEARNING_RATE = 0.001
    DEMO_RATIO = 0.1
    LAMBDA = [1.0, 0.0, 1.0, 10e-5]  # for [loss_dq, loss_n_dq, loss_jeq, loss_l2]
    PRETRAIN_STEPS = 1000  # 750000
    MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model/DQfD_model_mountaincar-0313')
    DEMO_DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'demo_M.p')
    MODEL_PATH_soul=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model/DQfD_model_CartPole-0322')

    demo_buffer_size = 696
    replay_buffer_size = demo_buffer_size *10
    iteration = 3
    episode = 100  # 300 games per iteration
    trajectory_n = 50  # for n-step TD-loss (both demo data and generated data)

    generate_memory_size = replay_buffer_size#demo_buffer_size
    expert_memory_size = replay_buffer_size

class DDQNConfig(Config):
    demo_mode = 'get_demo'
    demo_num = 0


class DQfDConfig(Config):
    demo_mode = 'use_demo'
    demo_num = int(Config.BATCH_SIZE * Config.DEMO_RATIO)



import os

class Config:
    ENV_NAME = "CartPole-v1" # "MountainCar-v0"  #"Seaquest-v0"#BipedalWalker-v2"#LunarLander-v2" #Acrobot-v1"#"MountainCar-v0"#"CartPole-v1"# "CartPole-v0"#"MountainCar-v0"#"#"Acrobot-v1"#
    GAMMA = 0.99
    DEMO_DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'demo.p')
    DEMO_BUFFER_SIZE = 9000
    BATCH_SIZE = 64
    trajectory_n = 50  # for n-step TD-loss (both demo data and generated data)
    #configures for DQFD

    AGENT_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model/agent_model_CartPole_0411_1')
    REPLAY_BUFFER_SIZE =18000
    PRETRAIN_STEPS = 1000
    UPDATE_TARGET_NET = 200

    LEARNING_RATE = 0.001
    DEMO_RATIO = 0.1
    LAMBDA = [1.0, 0.0, 1.0, 10e-5]  # for [loss_dq, loss_n_dq, loss_jeq, loss_l2]
    INITIAL_EPSILON = 1.0
    EPSILIN_DECAY = 0.999
    FINAL_EPSILON = 0.01  # final value of epsilon
    #configures for GAIL
    SOUL_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model/soul_model_CartPole_0411_2')
    EXPERT_MEMORY_SIZE = 9000
    GENERATE_MEMORY_SIZE = 9000

    #configures for ppo
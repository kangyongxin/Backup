"""
test for some basic command about your env
"""
import gym
import numpy as np
from baselines.UpTrend.maze_env import Maze
from baselines.MuscleMemory.RL_brain import QLearningTable,InternalModel

def random_policy():
    act = env.action_space.sample()
    return act

def main_MR():
    n_trj = 10
    for eps in range(n_trj):
        env.reset()
        while True:
            act = random_policy()
            obs, reward, done, _ = env.step(act)
            env.render()
            if done:
                print("done!")
                break

def main_MAZE():
    n_trj = 10
    RL = QLearningTable(actions=list(range(env.n_actions)))
    for eps in range(n_trj):
        observation = env.reset()
        while True:
            env.render()
            action = RL.random_action(str(observation))
            observation_, reward, done = env.step(action)
            observation = observation_

            if done:
                print("done!")
                break


if __name__ == "__main__":
    # env = gym.make('MontezumaRevengeNoFrameskip-v4')
    # main_MR()
    env = Maze()
    main_MAZE()
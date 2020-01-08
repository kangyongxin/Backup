"""
This part of code is the Q learning brain, which is a brain of the agent.
All decisions are made in here.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/
"""

import numpy as np
import pandas as pd


class QLearningTable:
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions  # a list
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    def choose_action(self, observation):
        self.check_state_exist(observation)
        # action selection
        if np.random.uniform() < self.epsilon:
            # choose best action
            state_action = self.q_table.loc[observation, :]
            # some actions may have the same value, randomly choose on in these actions
            action = np.random.choice(state_action[state_action == np.max(state_action)].index)
        else:
            # choose random action
            action = np.random.choice(self.actions)
        return action
    def random_action(self,observation):
        action = np.random.choice(self.actions)
        return action

    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.loc[s_, :].max()  # next state is not terminal
        else:
            q_target = r  # next state is terminal
        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)  # update

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(
                pd.Series(
                    [0]*len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )


class InternalModel(object):
    """
    Description:
        We'll create a tabular model for our simulated experience. Please complete the following code.
    """

    def __init__(self):
        # self.model = dict()
        self.model = []
        self.rand = np.random

    def store(self, state, action, next_state, reward):
        """
        TODO:
            Store the previous experience into the model.
        Return:
            NULL
        """
        exp = (state, action, next_state, reward)
        self.model.append(exp)

    def sample(self):
        """
        TODO:
            Randomly sample previous experience from internal model.
        Return:
            state, action, next_state, reward
        """
        import random
        (stateP, actionP, next_stateP, rewardP) = random.sample(self.model, 1)[0]
        return (stateP, actionP, next_stateP, rewardP)
    def check(self,state, action, next_state, reward):
        exp = (state, action, next_state, reward)
        if exp in self.model:
            return True
        else:
            return False
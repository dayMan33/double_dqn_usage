import numpy as np
from double_dqn.dqn_env import DQNEnv


class SimpleEnv(DQNEnv):
    """
    A very simple environment in which a state is a vector of n integers in the range(low, high). Each action swaps 2
    coordinates and a state is terminal if the list is sorted. The maximal number of swaps required is n, so the
    environment resets if more than (50 + n^2) actions were taken"""

    def __init__(self, n, low, high):
        """
        C-tor
        :param n: length of the vector representing the state.
        :param low: lower bound to draw integers from.
        :param high: upper bound to draw integers from.
        """
        DQNEnv.__init__(self)
        self._n = n
        self._low = low
        self._high = high
        self.state = None
        self.counter = 0
        self.max_actions = 50 + (n ** 2)
        self.actions = [[i, j] for i in range(n) for j in range(i + 1, n)]
        self.reset()

    def get_legal_actions(self, states):
        # all actions are always legal, so return a boolean vector filled with True
        return np.ones(len(self.actions))

    def step(self, action):
        """
        Advances the state of the environment by taking the action specified
        :param action: action to take.
        :return: current state of the environment after taking the action, the reward observed and a boolean parameter
                 representing whether or not the current state is a terminal state.
        """
        self.counter += 1
        self.apply_action(action)

        state, reward, terminal = self.state, -1, False
        if self.is_terminal():
            reward, terminal = 0, True
        elif self.counter == self.max_actions:
            terminal = True
        return state, reward, terminal

    def apply_action(self, action):
        """
        Applies an action, swapping 2 coordinates in the environments state based on the action taken
        :param action: the action to take
        """
        temp = self.state.copy()
        temp[self.actions[action][0]] = temp[self.actions[action][1]]
        temp[self.actions[action][1]] = self.state[self.actions[action][0]]
        self.state = temp

    def reset(self):
        self.state = np.random.randint(self._low, self._high + 1, self._n)
        self.counter = 0
        if self.is_terminal():
            return self.reset()
        return self.state

    def is_terminal(self):
        """
        Check if the current state of the environment is terminal
        :return: True iff self.state is sorted
        """
        return np.all(self.state[:-1] <= self.state[1:])

    def get_state_shape(self):
        return self.state.shape

    def get_action_shape(self):
        return (len(self.actions),)

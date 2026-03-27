import numpy as np
from ..core import Policy


class RandomPolicy(Policy):
    """Uniform random policy over discrete actions."""

    def __init__(self, n_actions: int):
        self.n_actions = n_actions

    def __call__(self, obs):
        return np.random.randint(self.n_actions)

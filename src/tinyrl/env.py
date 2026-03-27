from abc import ABC, abstractmethod
import numpy as np


class Environment(ABC):
    """
    Base class for RL environments.

    Subclasses must implement:
        reset()          -> initial observation
        step(action)     -> (observation, reward, done)
        _get_obs()       -> current observation as numpy array

    Subclasses must set in __init__:
        state_dim   (int): dimensionality of the observation
        n_actions   (int): number of discrete actions
        max_steps   (int): episode step limit
    """

    state_dim: int
    n_actions: int
    max_steps: int

    @abstractmethod
    def reset(self) -> np.ndarray:
        """Reset the environment and return the initial observation."""
        ...

    @abstractmethod
    def step(self, action: int) -> tuple[np.ndarray, float, bool]:
        """
        Take an action in the environment.

        Returns:
            observation: current state after the action
            reward: scalar reward signal
            done: whether the episode has ended
        """
        ...

    @abstractmethod
    def _get_obs(self) -> np.ndarray:
        """Return the current observation as a numpy array."""
        ...

    @abstractmethod
    def render(self, action: int | None = None, step_num: int = 0):
        """Display the current state of the environment."""
        ...

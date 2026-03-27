from abc import ABC, abstractmethod
import numpy as np
from .types import PolicyOutput


class Policy(ABC):
    """
    Base class for policies.

    Subclass this and implement __call__ to define a policy that works
    with Runner.run_episode().

    Your __call__ should return either:
        - an int (discrete action)
        - a np.ndarray (continuous action)
        - a PolicyOutput (action + optional logprob, entropy)

    Example:
        class MyPolicy(Policy):
            def __call__(self, obs):
                action = my_model(obs)
                return PolicyOutput(action=action, logprob=-0.5, entropy=1.2)

        runner.run_episode(MyPolicy())
    """

    @abstractmethod
    def __call__(self, obs: np.ndarray) -> int | np.ndarray | PolicyOutput:
        ...

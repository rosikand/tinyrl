from dataclasses import dataclass
import numpy as np


@dataclass
class PolicyOutput:
    """
    Structured return type for policy functions.

    Only action is required. Everything else is optional and will be
    stored on the Step if provided.

    Usage:
        def policy(obs):
            # simple
            return PolicyOutput(action=2)

            # with training info
            return PolicyOutput(action=2, logprob=-0.3, entropy=1.2)
    """
    action: int | np.ndarray
    logprob: float | None = None
    entropy: float | None = None


@dataclass
class Step:
    """A single environment transition."""
    obs: np.ndarray
    action: int | np.ndarray
    reward: float
    next_obs: np.ndarray
    done: bool
    logprob: float | None = None
    entropy: float | None = None


@dataclass
class Trajectory:
    """A full episode trajectory."""
    steps: list[Step]
    total_reward: float
    length: int

    def __iter__(self):
        return iter(self.steps)

    def __len__(self):
        return len(self.steps)

    def __getitem__(self, idx):
        return self.steps[idx]

    @property
    def obs(self) -> np.ndarray:
        """Batched observations, shape (length, state_dim)."""
        return np.stack([s.obs for s in self.steps])

    @property
    def actions(self) -> np.ndarray:
        """Batched actions. Shape (length,) for discrete, (length, action_dim) for continuous."""
        return np.stack([s.action for s in self.steps])

    @property
    def rewards(self) -> np.ndarray:
        """Batched rewards, shape (length,)."""
        return np.array([s.reward for s in self.steps])

    @property
    def logprobs(self) -> np.ndarray | None:
        """Batched log-probs, shape (length,). None if policy didn't provide them."""
        if self.steps[0].logprob is None:
            return None
        return np.array([s.logprob for s in self.steps])

    @property
    def entropies(self) -> np.ndarray | None:
        """Batched entropies, shape (length,). None if policy didn't provide them."""
        if self.steps[0].entropy is None:
            return None
        return np.array([s.entropy for s in self.steps])

# Adding Environments

To create a new environment, subclass `Environment` and implement four methods.

## Template

```python
import numpy as np
from tinyrl import Environment


class MyEnv(Environment):
    def __init__(self):
        self.state_dim = 4        # observation dimensionality
        self.n_actions = 2        # for discrete action spaces
        # self.action_dim = 3     # for continuous action spaces
        self.max_steps = 200      # episode step limit
        # ... your state variables ...

    def reset(self) -> np.ndarray:
        """Reset to initial state, return first observation."""
        self.steps = 0
        # ... reset your state ...
        return self._get_obs()

    def step(self, action) -> tuple[np.ndarray, float, bool]:
        """Take action, return (observation, reward, done)."""
        self.steps += 1
        # ... apply action, compute reward ...
        done = self.steps >= self.max_steps  # or some goal condition
        return self._get_obs(), reward, done

    def _get_obs(self) -> np.ndarray:
        """Return current state as numpy array."""
        # ... build observation vector ...
        return np.array([...], dtype=np.float32)

    def render(self, action=None, step_num=0):
        """Print or display the current state."""
        print(f"Step {step_num}: {self._get_obs()}")
```

## Checklist

1. Set `state_dim` and `max_steps` in `__init__`
2. Set `n_actions` (discrete) or `action_dim` (continuous)
3. `reset()` returns the initial observation and resets all state
4. `step()` returns `(obs, reward, done)` — make sure `done=True` when the episode should end
5. `_get_obs()` returns a numpy array with shape `(state_dim,)`
6. `render()` displays something useful for debugging

## Using with Runner

Once implemented, your environment works with `Runner` automatically:

```python
from tinyrl import Runner

env = MyEnv()
runner = Runner(env)

result = runner.run_episode(my_policy, visualize=True)
runner.plot()
```

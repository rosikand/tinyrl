# tinyrl

A minimal reinforcement learning toolkit built from scratch. Provides simple RL environments and training utilities for learning and experimenting with RL algorithms. 

See the docs [here](https://rosikand.github.io/tinyrl/) to get started. 

*Built with the help of claude code and a little bit of "taste"* 

## Install

```bash
pip install git+https://github.com/rosikand/tinyrl.git
```

## Usage

```python
from tinyrl import GridWorld, Runner
import numpy as np

env = GridWorld()
runner = Runner(env)

# random policy
runner.run_episode(lambda obs: np.random.randint(env.n_actions))

# visualize an episode
runner.run_episode(lambda obs: np.random.randint(env.n_actions), visualize=True)

# plot training stats
runner.plot()
```

## Environments

- **GridWorld** — 5x5 grid, agent navigates from (0,0) to goal at (4,4). Actions: up, right, down, left. Reward: -1 per step, +10 at goal.

## Adding a new environment

Subclass `Environment` and implement `reset`, `step`, `_get_obs`, and `render`:

```python
from tinyrl import Environment

class MyEnv(Environment):
    def __init__(self):
        self.state_dim = ...
        self.n_actions = ...
        self.max_steps = ...

    def reset(self): ...
    def step(self, action): ...
    def _get_obs(self): ...
    def render(self, action=None, step_num=0): ...
```

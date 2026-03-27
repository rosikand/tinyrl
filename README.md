# `tinyrl`

> A minimal reinforcement learning toolkit built from scratch. Provides simple RL environments and training utilities for learning and experimenting with RL algorithms.

See the docs [here](https://rosikand.github.io/tinyrl/) to get started.

*Built with the help of claude code and a little bit of "taste"*

## Install

```bash
pip install git+https://github.com/rosikand/tinyrl.git
```

## Usage

```python
from tinyrl import GridWorld, Runner, RandomPolicy

env = GridWorld()
runner = Runner(env)
policy = RandomPolicy(n_actions=env.n_actions)

# run an episode
result = runner.run_episode(policy)
print(result.reward, result.steps)

# with trajectory
result = runner.run_episode(policy, return_trajectory=True)
print(result.trajectory.obs.shape)

# plot training stats
runner.plot()
```

## Package structure

```
tinyrl/
  core/          # Environment, Policy, Runner, TrainingMonitor, types
  envs/          # GridWorld, ...
  algorithms/    # RandomPolicy, ...
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
        self.n_actions = ...   # for discrete
        self.action_dim = ...  # for continuous
        self.max_steps = ...

    def reset(self): ...
    def step(self, action): ...
    def _get_obs(self): ...
    def render(self, action=None, step_num=0): ...
```

## Adding a new policy

Subclass `Policy` and implement `__call__`:

```python
from tinyrl import Policy, PolicyOutput

class MyPolicy(Policy):
    def __call__(self, obs):
        action = ...
        return PolicyOutput(action=action, logprob=-0.5, entropy=1.2)
```

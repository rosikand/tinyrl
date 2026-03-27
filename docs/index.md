# tinyrl

A minimal reinforcement learning toolkit built from scratch. Provides simple RL environments and training utilities for learning and experimenting with RL algorithms.

## Why tinyrl?

- **Simple** — small codebase, easy to read and modify
- **From scratch** — no dependency on OpenAI Gym or Gymnasium
- **Extensible** — add new environments by subclassing `Environment`, new policies by subclassing `Policy`
- **Batteries included** — comes with a `Runner` for rolling out episodes and a `TrainingMonitor` for plotting training curves

## Quick example

```python
from tinyrl import GridWorld, Runner, RandomPolicy

env = GridWorld()
runner = Runner(env)
policy = RandomPolicy(n_actions=env.n_actions)

# run 100 episodes
for _ in range(100):
    result = runner.run_episode(policy)

# plot reward and episode length curves
runner.plot()
```

## Package structure

```
tinyrl/
  core/          # Environment, Policy, Runner, TrainingMonitor, types
  envs/          # GridWorld, ...
  algorithms/    # RandomPolicy, ...
```

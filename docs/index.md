# tinyrl

A minimal reinforcement learning toolkit built from scratch. Provides simple RL environments and training utilities for learning and experimenting with RL algorithms.

## Why tinyrl?

- **Simple** — small codebase, easy to read and modify
- **From scratch** — no dependency on OpenAI Gym or Gymnasium
- **Extensible** — add new environments by subclassing `Environment`
- **Batteries included** — comes with a `Runner` for rolling out episodes and a `TrainingMonitor` for plotting training curves

## Quick example

```python
from tinyrl import GridWorld, Runner
import numpy as np

env = GridWorld()
runner = Runner(env)

# run 100 episodes with a random policy
for _ in range(100):
    runner.run_episode(lambda obs: np.random.randint(env.n_actions))

# plot reward, episode length, and entropy curves
runner.plot()
```

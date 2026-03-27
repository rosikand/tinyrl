# Getting Started

## Installation

```bash
pip install git+https://github.com/rosikand/tinyrl.git
```

Or clone and install in editable mode for development:

```bash
git clone https://github.com/rosikand/tinyrl.git
cd tinyrl
pip install -e .
```

## Your first episode

```python
from tinyrl import GridWorld, Runner
import numpy as np

env = GridWorld()
runner = Runner(env)

# random policy: pick a random action each step
random_policy = lambda obs: np.random.randint(env.n_actions)

# run and visualize one episode
runner.run_episode(random_policy, visualize=True)
```

## Training loop

A typical training loop looks like this:

```python
from tinyrl import GridWorld, Runner
import numpy as np

env = GridWorld()
runner = Runner(env)

for episode in range(500):
    # replace with your learned policy
    policy = lambda obs: np.random.randint(env.n_actions)
    reward, steps = runner.run_episode(policy)

    if episode % 100 == 0:
        print(f"Episode {episode}: reward={reward:.0f}, steps={steps}")

runner.plot()
```

The `Runner` automatically logs reward, episode length, and policy entropy to its internal `TrainingMonitor`. Call `runner.plot()` at any point to see training curves.

## Policy function

The `policy_fn` passed to `run_episode` can return either:

- **just an action**: `obs -> int`
- **action and entropy**: `obs -> (int, float)` — useful for tracking how the policy distribution changes during training

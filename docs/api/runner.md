# Runner

The `Runner` manages the episode loop, connecting an environment with a policy and an optional training monitor.

```python
from tinyrl import GridWorld, Runner

env = GridWorld()
runner = Runner(env)
```

## Constructor

### `Runner(env, monitor=None)`

**Args:**

- `env` — an `Environment` instance
- `monitor` — a `TrainingMonitor` instance. If `None`, a default one is created automatically.

## Methods

### `run_episode(policy_fn, visualize=False, delay=0.1, log=True, return_trajectory=False)`

Roll out a single episode.

**Args:**

- `policy_fn` — a callable that takes an observation and returns either:
    - `int` or `np.ndarray` — the action to take
    - `PolicyOutput` — action plus optional logprob and entropy
- `visualize` — if `True`, calls `env.render()` at each step
- `delay` — seconds between frames when visualizing
- `log` — if `True`, logs stats to the monitor. Set `False` for eval runs.
- `return_trajectory` — if `True`, populates `result.trajectory`

**Returns:** `EpisodeResult`

```python
# basic
result = runner.run_episode(policy)
result.reward       # total reward
result.steps        # number of steps

# with trajectory
result = runner.run_episode(policy, return_trajectory=True)
result.trajectory   # Trajectory object

# eval (no logging)
result = runner.run_episode(policy, log=False)
```

### `plot()`

Calls `self.monitor.plot()` to display training curves.

```python
for _ in range(500):
    runner.run_episode(policy)
runner.plot()
```

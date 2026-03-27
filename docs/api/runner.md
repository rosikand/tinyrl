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

### `run_episode(policy_fn, visualize=False, delay=0.1)`

Roll out a single episode.

**Args:**

- `policy_fn` — a callable that takes an observation and returns either:
    - `action` (int) — the action to take
    - `(action, entropy)` (tuple) — action plus the entropy of the policy distribution, for tracking
- `visualize` — if `True`, calls `env.render()` at each step
- `delay` — seconds between frames when visualizing

**Returns:** `(total_reward, steps)`

```python
# simple policy
reward, steps = runner.run_episode(lambda obs: 1)  # always go right

# policy that reports entropy
def my_policy(obs):
    action = model(obs)
    entropy = compute_entropy(model)
    return action, entropy

reward, steps = runner.run_episode(my_policy)
```

If the policy function doesn't return entropy, the runner assumes a uniform distribution (`ln(n_actions)`).

### `plot()`

Calls `self.monitor.plot()` to display training curves.

```python
for _ in range(500):
    runner.run_episode(policy)
runner.plot()
```

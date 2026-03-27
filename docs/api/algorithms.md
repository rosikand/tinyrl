# Algorithms

Built-in policy implementations.

## RandomPolicy

Uniform random policy over discrete actions.

```python
from tinyrl import GridWorld, Runner, RandomPolicy

env = GridWorld()
policy = RandomPolicy(n_actions=env.n_actions)
runner = Runner(env)

result = runner.run_episode(policy)
```

### Constructor

#### `RandomPolicy(n_actions: int)`

**Args:**

- `n_actions` — number of discrete actions to sample from uniformly

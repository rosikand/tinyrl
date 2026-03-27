# Policy

`tinyrl.Policy` is the abstract base class for policies. Subclass it to create reusable, structured policies that work with `Runner.run_episode()`.

```python
from tinyrl import Policy, PolicyOutput

class MyPolicy(Policy):
    def __call__(self, obs):
        action = ...
        return PolicyOutput(action=action, logprob=-0.5, entropy=1.2)
```

Using the `Policy` ABC is optional — `run_episode` also accepts any callable. The ABC is useful when building real policies because it provides a clear interface and catches missing implementations at instantiation time.

## Abstract methods

### `__call__(obs: np.ndarray) -> int | np.ndarray | PolicyOutput`

Choose an action given an observation.

**Args:**

- `obs` — current observation from the environment

**Returns:** one of:

- `int` — discrete action index
- `np.ndarray` — continuous action vector
- `PolicyOutput` — action with optional logprob and entropy

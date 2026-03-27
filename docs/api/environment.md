# Environment

`tinyrl.Environment` is the abstract base class for all environments. It defines the interface that every environment must implement.

## Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `state_dim` | `int` | Dimensionality of the observation vector |
| `n_actions` | `int \| None` | Number of discrete actions (for discrete envs) |
| `action_dim` | `int \| None` | Dimensionality of the action vector (for continuous envs) |
| `max_steps` | `int` | Maximum steps per episode |

Set `n_actions` for discrete action spaces, `action_dim` for continuous.

## Abstract methods

### `reset() -> np.ndarray`

Reset the environment to its initial state.

**Returns:** the initial observation as a numpy array.

```python
obs = env.reset()
```

### `step(action: int | np.ndarray) -> tuple[np.ndarray, float, bool]`

Take one step in the environment.

**Args:**

- `action` — action index (int, discrete) or action vector (np.ndarray, continuous)

**Returns:** a tuple of `(observation, reward, done)`

- `observation` — the new state as a numpy array
- `reward` — scalar reward for this transition
- `done` — `True` if the episode has ended

```python
obs, reward, done = env.step(action)
```

### `_get_obs() -> np.ndarray`

Return the current observation. Called internally by `reset()` and `step()`.

### `render(action=None, step_num=0)`

Display the current state of the environment.

**Args:**

- `action` — the action that was just taken (for display purposes)
- `step_num` — current step number in the episode

# Types

Data classes used throughout tinyrl.

## PolicyOutput

Structured return type for policy functions.

```python
from tinyrl import PolicyOutput

# just an action
PolicyOutput(action=2)

# with training info
PolicyOutput(action=2, logprob=-0.3, entropy=1.2)
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `action` | `int \| np.ndarray` | required | The action to take |
| `logprob` | `float \| None` | `None` | Log-probability of the action |
| `entropy` | `float \| None` | `None` | Entropy of the policy distribution |

## Step

A single environment transition, stored in trajectories.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `obs` | `np.ndarray` | required | Observation before the action |
| `action` | `int \| np.ndarray` | required | Action taken |
| `reward` | `float` | required | Reward received |
| `next_obs` | `np.ndarray` | required | Observation after the action |
| `done` | `bool` | required | Whether the episode ended |
| `logprob` | `float \| None` | `None` | Log-probability (if policy provided it) |
| `entropy` | `float \| None` | `None` | Entropy (if policy provided it) |

## Trajectory

A full episode trajectory. Supports indexing, iteration, and batched property access.

```python
result = runner.run_episode(policy, return_trajectory=True)
traj = result.trajectory

traj[0]            # first Step
len(traj)          # number of steps
for step in traj:  # iterate
    ...
```

### Batched properties

| Property | Shape | Description |
|----------|-------|-------------|
| `traj.obs` | `(length, state_dim)` | All observations |
| `traj.actions` | `(length,)` or `(length, action_dim)` | All actions |
| `traj.rewards` | `(length,)` | All rewards |
| `traj.logprobs` | `(length,)` or `None` | All log-probs (if provided) |
| `traj.entropies` | `(length,)` or `None` | All entropies (if provided) |

## EpisodeResult

Return type for `Runner.run_episode()`.

| Field | Type | Description |
|-------|------|-------------|
| `reward` | `float` | Total episode reward |
| `steps` | `int` | Number of steps taken |
| `trajectory` | `Trajectory \| None` | Full trajectory (if `return_trajectory=True`) |

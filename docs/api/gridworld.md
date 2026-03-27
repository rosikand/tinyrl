# GridWorld

A simple grid-based MDP. The agent starts at the top-left corner and must navigate to the bottom-right goal.

```python
from tinyrl import GridWorld

env = GridWorld(size=5)
```

## MDP specification

| Component | Value |
|-----------|-------|
| **States** | `(row, col)` positions, normalized to `[0, 1]` |
| **Actions** | `0`=up, `1`=right, `2`=down, `3`=left |
| **Transitions** | Deterministic. Moves in the chosen direction, clipped at walls |
| **Rewards** | `-1` per step, `+10` when reaching the goal |
| **Start** | `(0, 0)` — top-left |
| **Goal** | `(size-1, size-1)` — bottom-right |
| **Horizon** | 50 steps max |

## Constructor

### `GridWorld(size=5)`

**Args:**

- `size` — grid dimension (creates a `size x size` grid). Default: `5`.

## Example

```python
from tinyrl import GridWorld

env = GridWorld()
obs = env.reset()          # array([0., 0.])
obs, r, done = env.step(2) # move down -> array([0.25, 0.]), r=-1.0
obs, r, done = env.step(1) # move right -> array([0.25, 0.25]), r=-1.0
```

## Optimal policy

The shortest path from `(0,0)` to `(4,4)` takes 8 steps (4 down + 4 right), giving a total reward of `+2` (7 steps × -1 + 1 step with +10).

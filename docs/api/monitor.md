# TrainingMonitor

Logs episode statistics and plots training curves. Automatically used by `Runner`, but can also be used standalone.

```python
from tinyrl import TrainingMonitor

monitor = TrainingMonitor(window=50)
```

## Constructor

### `TrainingMonitor(window=50)`

**Args:**

- `window` — rolling average window size for smoothing plots

## Methods

### `log(reward, length, entropy)`

Record one episode's statistics.

**Args:**

- `reward` — total episode reward
- `length` — number of steps in the episode
- `entropy` — mean policy entropy over the episode

### `plot()`

Display three training curves side by side:

1. **Episode Reward** — with a reference line at the optimal reward
2. **Episode Length** — with a reference line at the optimal path length
3. **Policy Entropy** — with a reference line at maximum entropy (uniform policy)

Each plot shows the raw data (transparent) and a rolling average (solid line).

## Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `rewards` | `list[float]` | All logged episode rewards |
| `lengths` | `list[int]` | All logged episode lengths |
| `entropies` | `list[float]` | All logged mean entropies |

## Standalone usage

```python
from tinyrl import TrainingMonitor

monitor = TrainingMonitor()

for episode in range(500):
    # ... your training loop ...
    monitor.log(total_reward, steps, entropy)

monitor.plot()
```

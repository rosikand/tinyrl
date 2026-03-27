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

### `log(reward, length, entropy=None)`

Record one episode's statistics.

**Args:**

- `reward` — total episode reward
- `length` — number of steps in the episode
- `entropy` — mean policy entropy over the episode (optional)

### `plot()`

Display training curves side by side:

1. **Episode Reward**
2. **Episode Length**
3. **Policy Entropy** — only shown if entropy data was logged

Each plot shows the raw data (transparent) and a rolling average (solid line).

## Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `rewards` | `list[float]` | All logged episode rewards |
| `lengths` | `list[int]` | All logged episode lengths |
| `entropies` | `list[float]` | Logged mean entropies (only for episodes that provided entropy) |

## Standalone usage

```python
from tinyrl import TrainingMonitor

monitor = TrainingMonitor()

for episode in range(500):
    # ... your training loop ...
    monitor.log(total_reward, steps, entropy)  # entropy is optional

monitor.plot()
```

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
from tinyrl import GridWorld, Runner, RandomPolicy

env = GridWorld()
runner = Runner(env)
policy = RandomPolicy(n_actions=env.n_actions)

# run and visualize one episode
result = runner.run_episode(policy, visualize=True)
print(f"Reward: {result.reward}, Steps: {result.steps}")
```

## Training loop

A typical training loop looks like this:

```python
from tinyrl import GridWorld, Runner

env = GridWorld()
runner = Runner(env)

for episode in range(500):
    # replace with your learned policy
    result = runner.run_episode(my_policy)

    if episode % 100 == 0:
        print(f"Episode {episode}: reward={result.reward:.0f}, steps={result.steps}")

runner.plot()
```

The `Runner` automatically logs reward, episode length, and policy entropy (if provided) to its internal `TrainingMonitor`. Call `runner.plot()` at any point to see training curves.

## Policy function

The `policy_fn` passed to `run_episode` can be:

- **A plain callable** returning an `int` (discrete) or `np.ndarray` (continuous):
  ```python
  lambda obs: np.random.randint(4)
  ```

- **A callable returning `PolicyOutput`** with optional logprob and entropy:
  ```python
  from tinyrl import PolicyOutput

  def my_policy(obs):
      return PolicyOutput(action=2, logprob=-0.5, entropy=1.2)
  ```

- **A `Policy` subclass** for structured, reusable policies:
  ```python
  from tinyrl import Policy, PolicyOutput

  class MyPolicy(Policy):
      def __call__(self, obs):
          return PolicyOutput(action=2, entropy=1.2)
  ```

## EpisodeResult

`run_episode` always returns an `EpisodeResult`:

```python
result = runner.run_episode(policy)
result.reward       # total episode reward
result.steps        # number of steps taken
result.trajectory   # None unless return_trajectory=True
```

## Trajectories

Request the full trajectory to inspect or use for training:

```python
result = runner.run_episode(policy, return_trajectory=True)
traj = result.trajectory

traj[0].obs          # first step observation
traj.obs             # batched obs array, shape (steps, state_dim)
traj.actions         # batched actions
traj.rewards         # batched rewards
traj.logprobs        # batched logprobs (None if not provided)
traj.entropies       # batched entropies (None if not provided)
```

## Eval mode

Run episodes without logging to the monitor:

```python
result = runner.run_episode(policy, log=False)
```

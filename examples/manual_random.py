"""Custom policy using the Policy ABC and PolicyOutput."""

from tinyrl import GridWorld, Runner, Policy, PolicyOutput
import numpy as np


class ManualRandom(Policy):
    def __call__(self, obs):
        rand_step_param = np.random.randint(60)
        if rand_step_param > 45:
            action = 3
        elif rand_step_param > 30:
            action = 2
        elif rand_step_param > 15:
            action = 1
        else:
            action = 0

        probs = np.array([16/60, 15/60, 15/60, 14/60], dtype=np.float32)
        entropy = float(-(probs * np.log(probs)).sum())

        return PolicyOutput(action=action, entropy=entropy)


env = GridWorld()
runner = Runner(env)
policy = ManualRandom()

# train
for _ in range(200):
    runner.run_episode(policy)

# eval (no logging)
result = runner.run_episode(policy, log=False)
print(f"Eval: reward={result.reward:.0f}, steps={result.steps}")

# inspect a trajectory
result = runner.run_episode(policy, return_trajectory=True)
traj = result.trajectory
print(f"Trajectory: {len(traj)} steps, reward={result.reward:.0f}")
print(f"  obs batch shape: {traj.obs.shape}")
print(f"  mean entropy: {traj.entropies.mean():.3f}")

runner.plot()

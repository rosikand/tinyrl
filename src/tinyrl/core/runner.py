import time
import numpy as np
from .env import Environment
from .monitor import TrainingMonitor
from .types import PolicyOutput, Step, Trajectory, EpisodeResult


class Runner:
    """
    Runs episodes on an environment with an optional monitor.

    Usage:
        runner = Runner(env)
        runner.run_episode(policy_fn)
        runner.run_episode(policy_fn, visualize=True)
        runner.plot()
    """

    def __init__(self, env: Environment, monitor: TrainingMonitor | None = None):
        self.env = env
        self.monitor = monitor or TrainingMonitor()

    def run_episode(self, policy_fn, visualize=False, delay=0.1,
                    log=True, return_trajectory=False):
        """
        Roll out one episode.

        Args:
            policy_fn: callable, obs -> int | PolicyOutput
            visualize: if True, calls env.render() each step
            delay: seconds between frames when visualizing
            log: if True, logs stats to the monitor (set False for eval runs)
            return_trajectory: if True, returns the full trajectory

        Returns:
            EpisodeResult with reward, steps, and optionally trajectory.
        """
        obs = self.env.reset()
        if visualize:
            self.env.render()
            time.sleep(delay)

        done = False
        total_reward = 0.0
        steps = 0
        entropies = []
        trajectory = [] if return_trajectory else None

        while not done:
            result = policy_fn(obs)

            if isinstance(result, PolicyOutput):
                action = result.action
                logprob = result.logprob
                entropy = result.entropy
            else:
                action = result
                logprob = None
                entropy = None

            if entropy is not None:
                entropies.append(entropy)

            next_obs, reward, done = self.env.step(action)

            if return_trajectory:
                trajectory.append(Step(
                    obs, action, reward, next_obs, done,
                    logprob=logprob, entropy=entropy,
                ))

            obs = next_obs
            total_reward += reward
            steps += 1

            if visualize:
                self.env.render(action, steps)
                time.sleep(delay)

        mean_entropy = np.mean(entropies) if entropies else None
        if log:
            self.monitor.log(total_reward, steps, mean_entropy)

        traj = Trajectory(trajectory, total_reward, steps) if return_trajectory else None
        return EpisodeResult(reward=total_reward, steps=steps, trajectory=traj)

    def plot(self):
        self.monitor.plot()

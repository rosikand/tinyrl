import time
import numpy as np
from .env import Environment
from .monitor import TrainingMonitor


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

    def run_episode(self, policy_fn, visualize=False, delay=0.1):
        """
        Roll out one episode.

        Args:
            policy_fn: callable, obs -> action or obs -> (action, entropy)
            visualize: if True, calls env.render() each step
            delay: seconds between frames when visualizing

        Returns:
            (total_reward, steps)
        """
        obs = self.env.reset()
        if visualize:
            self.env.render()
            time.sleep(delay)

        done = False
        total_reward = 0.0
        steps = 0
        entropies = []

        while not done:
            result = policy_fn(obs)
            if isinstance(result, tuple):
                action, entropy = result
                entropies.append(entropy)
            else:
                action = result
                entropies.append(np.log(self.env.n_actions))

            obs, reward, done = self.env.step(action)
            total_reward += reward
            steps += 1

            if visualize:
                self.env.render(action, steps)
                time.sleep(delay)

        self.monitor.log(total_reward, steps, np.mean(entropies))
        return total_reward, steps

    def plot(self):
        self.monitor.plot()

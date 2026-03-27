"""Simplest possible example: a random policy on GridWorld."""

from tinyrl import GridWorld, Runner
import numpy as np

env = GridWorld()
runner = Runner(env)

for _ in range(200):
    runner.run_episode(lambda obs: np.random.randint(env.n_actions))

runner.plot()

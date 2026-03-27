import numpy as np
from IPython.display import clear_output
from ..core import Environment


class GridWorld(Environment):
    """
    5x5 grid MDP.
      - States:  (row, col) positions, normalized to [0,1].
      - Actions: 0=up, 1=right, 2=down, 3=left
      - Rewards: -1 per step, +10 at goal.
      - Horizon: 50 steps max.

    Agent starts at (0,0), goal is (size-1, size-1).
    """

    def __init__(self, size=5):
        self.size = size
        self.goal = (size - 1, size - 1)
        self.state_dim = 2
        self.n_actions = 4
        self.max_steps = 50
        self.pos = [0, 0]
        self.steps = 0

    def reset(self):
        self.pos = [0, 0]
        self.steps = 0
        return self._get_obs()

    def _get_obs(self):
        return np.array(self.pos, dtype=np.float32) / (self.size - 1)

    def step(self, action):
        self.steps += 1
        moves = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        dr, dc = moves[action]
        self.pos[0] = np.clip(self.pos[0] + dr, 0, self.size - 1)
        self.pos[1] = np.clip(self.pos[1] + dc, 0, self.size - 1)

        done = tuple(self.pos) == self.goal or self.steps >= self.max_steps
        reward = 10.0 if tuple(self.pos) == self.goal else -1.0
        return self._get_obs(), reward, done

    def render(self, action=None, step_num=0):
        action_names = ['up', 'right', 'down', 'left']
        clear_output(wait=True)

        header = f"Step: {step_num}"
        if action is not None:
            header += f"  |  Action: {action_names[action]}"
        print(header)
        print("+" + "---+" * self.size)

        for r in range(self.size):
            row = "|"
            for c in range(self.size):
                if [r, c] == self.pos:
                    row += " A |"
                elif (r, c) == self.goal:
                    row += " G |"
                else:
                    row += "   |"
            print(row)
            print("+" + "---+" * self.size)

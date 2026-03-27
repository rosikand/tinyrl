import numpy as np
import matplotlib.pyplot as plt


class TrainingMonitor:
    """Logs and plots episode rewards, lengths, and policy entropy."""

    def __init__(self, window=50):
        self.rewards = []
        self.lengths = []
        self.entropies = []
        self.window = window

    def log(self, reward, length, entropy):
        self.rewards.append(reward)
        self.lengths.append(length)
        self.entropies.append(entropy)

    def _smooth(self, data):
        if len(data) < self.window:
            return data
        return np.convolve(data, np.ones(self.window) / self.window, mode='valid')

    def plot(self):
        fig, axes = plt.subplots(1, 3, figsize=(16, 4))
        for ax, data, title, color, ylabel in zip(
            axes,
            [self.rewards, self.lengths, self.entropies],
            ['Episode Reward', 'Episode Length', 'Policy Entropy'],
            ['#2196F3', '#4CAF50', '#FF9800'],
            ['Reward', 'Steps', 'H(pi)']
        ):
            ax.plot(data, alpha=0.2, color=color)
            smoothed = self._smooth(data)
            offset = len(data) - len(smoothed)
            ax.plot(range(offset, len(data)), smoothed, color=color, linewidth=2)
            ax.set_title(title)
            ax.set_xlabel('Episode')
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)

        axes[0].axhline(y=2, color='red', linestyle='--', alpha=0.5, label='Optimal (+2)')
        axes[0].legend()
        axes[1].axhline(y=8, color='red', linestyle='--', alpha=0.5, label='Optimal (8 steps)')
        axes[1].legend()
        axes[2].axhline(y=np.log(4), color='red', linestyle='--', alpha=0.5, label='Max entropy (uniform)')
        axes[2].legend()
        plt.tight_layout()
        plt.show()

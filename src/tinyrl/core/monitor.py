import numpy as np
import matplotlib.pyplot as plt


class TrainingMonitor:
    """Logs and plots episode rewards, lengths, and (optionally) policy entropy."""

    def __init__(self, window=50):
        self.rewards = []
        self.lengths = []
        self.entropies = []
        self.window = window

    def log(self, reward, length, entropy=None):
        self.rewards.append(reward)
        self.lengths.append(length)
        if entropy is not None:
            self.entropies.append(entropy)

    def _smooth(self, data):
        if len(data) < self.window:
            return data
        return np.convolve(data, np.ones(self.window) / self.window, mode='valid')

    def _plot_panel(self, ax, data, title, color, ylabel, ref_y=None, ref_label=None):
        ax.plot(data, alpha=0.2, color=color)
        smoothed = self._smooth(data)
        offset = len(data) - len(smoothed)
        ax.plot(range(offset, len(data)), smoothed, color=color, linewidth=2)
        ax.set_title(title)
        ax.set_xlabel('Episode')
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        if ref_y is not None:
            ax.axhline(y=ref_y, color='red', linestyle='--', alpha=0.5, label=ref_label)
            ax.legend()

    def plot(self):
        has_entropy = len(self.entropies) > 0
        n_panels = 3 if has_entropy else 2

        fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 4))

        self._plot_panel(axes[0], self.rewards, 'Episode Reward', '#2196F3', 'Reward')
        self._plot_panel(axes[1], self.lengths, 'Episode Length', '#4CAF50', 'Steps')

        if has_entropy:
            self._plot_panel(axes[2], self.entropies, 'Policy Entropy', '#FF9800', 'H(pi)')

        plt.tight_layout()
        plt.show()

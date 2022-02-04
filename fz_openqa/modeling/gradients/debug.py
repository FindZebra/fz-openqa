import numpy as np
import rich
import torch


@torch.no_grad()
def plot_scores(scores, controlled_scores):
    import seaborn as sns
    import matplotlib.pyplot as plt

    sns.set()
    colors = sns.color_palette()

    rich.print(f">> scores: {scores.shape}")
    rich.print(f">> controlled_scores: {controlled_scores.shape}")

    scores = scores.detach().cpu().numpy()
    controlled_scores = controlled_scores.detach().cpu().numpy()

    for k in range(scores.shape[0]):
        rich.print(f">> score: {scores[k]}, controlled_score: {controlled_scores[k]}")

    sns.distplot(scores, bins=10, color=colors[0], label="score")
    sns.distplot(controlled_scores, bins=10, color=colors[1], label="controlled")
    plt.axvline(x=np.mean(scores), color=colors[0])
    plt.axvline(x=np.mean(controlled_scores), color=colors[1])
    plt.legend()
    plt.show()

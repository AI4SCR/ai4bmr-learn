import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import wandb


def log_class_distribution(records: list, metadata: dict = None):
    metadata = metadata or {}
    for item in records:
        fig, ax = plt.subplots()

        name = item["name"]
        labels = item["labels"]
        y_true = item["y_true"]

        pdat = pd.DataFrame(dict(value=y_true))
        sns.countplot(data=pdat, x="value", ax=ax)

        ax.set_xticks(ax.get_xticks(), labels)
        ax.set_title(f"Class distribution")

        wandb.log({name: wandb.Image(fig), **metadata})
        plt.close(fig)

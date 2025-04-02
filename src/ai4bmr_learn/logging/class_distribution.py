import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import wandb


def log_class_distribution(records: list, labels: list, outer_fold: int = 0):
    for panel, true, pred in records:
        fig, ax = plt.subplots()

        pdat = pd.DataFrame(dict(value=true))
        sns.countplot(data=pdat, x="value", ax=ax)

        ax.set_xticks(ax.get_xticks(), labels)
        ax.set_title("Class distribution")

        wandb.log({f"class_distribution/{panel}": wandb.Image(fig), "outer_fold": outer_fold})
        plt.close(fig)

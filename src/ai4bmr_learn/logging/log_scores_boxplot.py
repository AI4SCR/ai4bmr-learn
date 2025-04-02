from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import wandb


def log_scores_boxplot(records: list):
    for panel, data in records:
        fig, ax = plt.subplots()

        pdat = pd.DataFrame.from_records(data)
        pdat = pdat.melt(id_vars="outer_fold")
        pdat["value"] = pdat.value.astype(float)

        sns.boxplot(data=pdat, x="variable", y="value", ax=ax)
        sns.stripplot(data=pdat, x="variable", y="value", hue="outer_fold", ax=ax)

        ax.set_title(panel)

        wandb.log({f"scores/{panel}": wandb.Image(fig)})
        plt.close(fig)

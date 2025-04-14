import seaborn as sns
import wandb
from ai4bmr_core.utils.plotting import get_grid_dims
from matplotlib import pyplot as plt


def log_scores_boxplot(records: list, metadata: dict = None):
    metadata = metadata or {}

    for item in records:

        name, scores = item["name"], item["scores"]
        pdat = scores.melt(id_vars="outer_fold")
        pdat["value"] = pdat.value.astype(float)

        num_vars = pdat.variable.nunique()
        nrows, ncols = get_grid_dims(num_vars)
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 5, nrows * 3))

        for ax, (grp_name, grp_data) in zip(axs.flat, pdat.groupby("variable")):
            sns.boxplot(data=grp_data, x="variable", y="value", ax=ax)
            sns.stripplot(data=grp_data, x="variable", y="value", hue="outer_fold", ax=ax)
            ax.set_title(grp_name)

        for ax in axs.flatten()[num_vars:]:
            ax.set_axis_off()

        fig.tight_layout()
        # fig.show()

        wandb.log({name: wandb.Image(fig), **metadata})
        plt.close(fig)

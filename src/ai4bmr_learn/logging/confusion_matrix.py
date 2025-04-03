from sklearn.metrics import ConfusionMatrixDisplay
from matplotlib import pyplot as plt
import wandb


def log_confusion_matrix(records: list, metadata: dict = None):
    metadata = metadata or {}
    for item in records:
        split = item["split"]
        labels = item["labels"]
        y_true, y_pred = item["y_true"], item["y_pred"]

        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        fig.suptitle(split)

        axs[0].set_title(f"count")
        display = ConfusionMatrixDisplay.from_predictions(
            y_true, y_pred, normalize=None, display_labels=labels, ax=axs[0], cmap="Blues"
        )

        axs[1].set_title(f"frequency")
        display = ConfusionMatrixDisplay.from_predictions(
            y_true, y_pred, normalize="true", display_labels=labels, ax=axs[1], cmap="Blues"
        )
        fig.tight_layout()

        wandb.log({f"confusion_matrix/{split}": wandb.Image(fig), **metadata})

        plt.close(fig)

from sklearn.metrics import ConfusionMatrixDisplay
from matplotlib import pyplot as plt
import wandb


def log_confusion_matrix(records: list, labels: list, outer_fold: int = 0):
    for panel, y_true, y_pred in records:
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        fig.suptitle(panel)

        axs[0].set_title(f"count")
        display = ConfusionMatrixDisplay.from_predictions(
            y_true, y_pred, normalize=None, display_labels=labels, ax=axs[0], cmap="Blues"
        )

        axs[1].set_title(f"frequency")
        display = ConfusionMatrixDisplay.from_predictions(
            y_true, y_pred, normalize="true", display_labels=labels, ax=axs[1], cmap="Blues"
        )
        fig.tight_layout()
        wandb.log({f"confusion_matrix/{panel}": wandb.Image(fig), "outer_fold": outer_fold})

        plt.close(fig)

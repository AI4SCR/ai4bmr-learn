# %%
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import seaborn as sns
import torch
import wandb
from ai4bmr_learn.data_models.WandInitConfig import WandbInitConfig
from ai4bmr_learn.datamodules.MIL import MILDataModule
from ai4bmr_learn.models.mil.ABMIL import ABMILModule
from ai4bmr_learn.train.train import TrainerConfig, get_trainer
from lightning import seed_everything
from matplotlib import pyplot as plt
from torchmetrics import MetricCollection

from ..metrics.classification import get_metric_collection


# %%
def abmil(
    datamodule: MILDataModule,
    weighted: bool = False,
    trainer: TrainerConfig = TrainerConfig(),
    wandb_init: WandbInitConfig = WandbInitConfig(),
    metric_collection: MetricCollection = None,
):
    # DATA
    dm = datamodule
    dm.prepare_data()
    dm.setup()

    labels = dm.dataset.labels

    if weighted:
        class_distribution = dm.dataset.class_distribution
        class_weight = 1 / class_distribution
        class_weight /= class_weight.sum()
    else:
        class_weight = None

    # METRICS
    metric_collection = metric_collection or get_metric_collection(num_classes=dm.dataset.num_classes)
    metrics_train = metric_collection.clone(prefix="abmil:train/")
    metrics_test = metrics_train.clone(prefix="abmil:test/")

    # MODEL
    module = ABMILModule(
        num_classes=dm.dataset.num_classes, num_features=dm.dataset.num_features, class_weight=class_weight
    )

    # RUN CONFIGURATION
    wandb.init(**asdict(wandb_init))

    # METADATA
    ckpt_dir = Path(wandb.run.dir) / "checkpoints" / wandb_init.project / wandb.run.name

    # TRAIN
    trainer = get_trainer(config=trainer, ckpt_dir=ckpt_dir)
    seed_everything(42, workers=True)
    trainer.fit(model=module, datamodule=dm)

    # BEST MODEL
    ckpt_path = trainer.checkpoint_callback.best_model_path
    best_model = ABMILModule.load_from_checkpoint(ckpt_path)

    # PREDICTIONS
    test_results = trainer.predict(model=best_model, dataloaders=dm.test_dataloader())
    y_test_pred = torch.concat([i["prediction"] for i in test_results])
    y_test = torch.concat([i["target"] for i in test_results])

    train_results = trainer.predict(model=best_model, dataloaders=dm.train_dataloader())
    y_train_pred = torch.concat([i["prediction"] for i in train_results])
    y_train = torch.concat([i["target"] for i in train_results])

    # TEST SCORES
    scores_test = metrics_test(y_test_pred, y_test)
    # scores_test["outer_fold"] = outer_fold
    wandb.log(scores_test)
    # overall_test.append(scores_test)

    # TRAIN SCORES
    scores_train = metrics_train(y_train_pred, y_train)
    # scores_train["outer_fold"] = outer_fold
    wandb.log(scores_train)
    # overall_train.append(scores_train)

    # VISUALIZATIONS
    # 1. Class distribution
    for panel, data in [("train", y_train), ("test", y_test)]:
        fig, ax = plt.subplots()
        pdat = pd.DataFrame(dict(value=data))
        sns.countplot(data=pdat, x="value", ax=ax)
        ax.set_xticks(ax.get_xticks(), labels)
        ax.set_title("Class distribution")
        # wandb.log({f"class_distribution/{panel}": wandb.Image(fig), "outer_fold": outer_fold})
        wandb.log({f"class_distribution/{panel}": wandb.Image(fig)})
        plt.close(fig)

    # 2. Confusion matrix
    from sklearn.metrics import ConfusionMatrixDisplay

    for panel, y_true, y_pred in [("train", y_train, y_train_pred), ("test", y_test, y_test_pred)]:
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
        # wandb.log({f"confusion_matrix/{panel}": wandb.Image(fig), "outer_fold": outer_fold})
        wandb.log({f"confusion_matrix/{panel}": wandb.Image(fig)})
        plt.close(fig)

    wandb.finish()

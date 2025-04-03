# %%
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd
import seaborn as sns
import torch
import wandb
from ai4bmr_learn.data_models.WandInitConfig import WandbInitConfig
from ai4bmr_learn.models.mil.ABMIL import ABMILModule
from ai4bmr_learn.train.train import TrainerConfig, get_trainer
from ai4bmr_learn.datamodules.MIL import MILDataModule, MILDataModuleConfig

from lightning import seed_everything
from matplotlib import pyplot as plt
from torchmetrics import Accuracy, F1Score, MetricCollection, Precision, Recall


@dataclass
class Parameters:
    # TODO: add more tunable parameters
    gated: bool = False


@dataclass
class SweepConfig:
    # method: str = "grid"
    num_outer_cv: int = 2
    # parameters: Parameters = field(default_factory=lambda: Parameters())


# %%


def main(
    datamodule: MILDataModule,
    trainer: TrainerConfig = TrainerConfig(),
    wandb_init: WandbInitConfig = WandbInitConfig(),
):

    # DATA
    outer_fold = 0
    splits_path = Path(datamodule.metadata_path.parent / "splits" / f"fold_{outer_fold}.parquet")
    splits_path.parent.mkdir(parents=True, exist_ok=True)
    datamodule.splits_path = splits_path

    dm = datamodule
    dm.prepare_data()
    dm.setup()

    labels = dm.dataset.labels

    # METRICS
    task = "multiclass" if dm.num_classes > 2 else "binary"
    metrics_train = MetricCollection(
        {
            "accuracy": Accuracy(task=task, num_classes=dm.num_classes),
            "recall": Recall(task=task, num_classes=dm.num_classes),
            "precision": Precision(task=task, num_classes=dm.num_classes),
            "f1": F1Score(task=task, num_classes=dm.num_classes),
        },
        prefix="train/",
    )
    metrics_test = metrics_train.clone(prefix="test/")

    # MODEL
    module = ABMILModule(num_classes=dm.num_classes, num_features=dm.num_features)

    # METADATA
    metadata = dict(ckpt_dir=Path(wandb.run.dir) / "checkpoints" / f"fold_{outer_fold}")

    # RUN CONFIGURATION
    wandb.init(**asdict(wandb_init), config=metadata)

    # TRAIN
    trainer = get_trainer(config=trainer, metadata=metadata)
    seed_everything(42, workers=True)
    trainer.fit(model=module, datamodule=dm)

    # BEST MODEL
    ckpt_path = trainer.checkpoint_callback.best_model_path
    best_model = ABMILModule.load_from_checkpoint(ckpt_path)

    # PREDICTIONS
    test_results = trainer.predict(model=best_model, dataloaders=dm.test_dataloader())
    y_test_pred, y_test = test_results["predictions"], test_results["targets"]

    train_results = trainer.predict(model=best_model, dataloaders=dm.train_dataloader())
    y_train_pred, y_train = train_results["predictions"], train_results["targets"]

    # TEST SCORES
    scores_test = metrics_test(y_test_pred, y_test)
    scores_test["outer_fold"] = outer_fold
    wandb.log(scores_test)
    # overall_test.append(scores_test)

    # TRAIN SCORES
    scores_train = metrics_train(y_train_pred, y_train)
    scores_train["outer_fold"] = outer_fold
    wandb.log(scores_train)
    # overall_train.append(scores_train)

    # RESET (not sure if this necessary if we only use the return values of the metrics object)
    metrics_train.reset()
    metrics_test.reset()

    # VISUALIZATIONS
    # 1. Class distribution
    for panel, data in [("train", y_train), ("test", y_test)]:
        fig, ax = plt.subplots()
        pdat = pd.DataFrame(dict(value=data))
        sns.countplot(data=pdat, x="value", ax=ax)
        ax.set_xticks(ax.get_xticks(), labels)
        ax.set_title("Class distribution")
        wandb.log({f"class_distribution/{panel}": wandb.Image(fig), "outer_fold": outer_fold})
        plt.close(fig)

    # 2. Confusion matrix
    from sklearn.metrics import ConfusionMatrixDisplay

    for panel, y_true, y_pred in [("train", y_train, y_train_pred), ("test", y_test, y_test_pred)]:
        fig, ax = plt.subplots()
        ax.set_title(panel)
        display = ConfusionMatrixDisplay.from_predictions(
            y_true, y_pred, normalize="true", display_labels=labels, ax=ax, cmap="Blues"
        )
        wandb.log({f"confusion_matrix/{panel}": wandb.Image(fig), "outer_fold": outer_fold})
        plt.close(fig)

    # NOTE: experiment with native plotting functions, keep for now as doc
    # from wandb.sklearn import plot_class_proportions
    # plot_class_proportions(y_train, y_test, labels)

    # cm = wandb.plot.confusion_matrix(
    #     y_true=y_test.numpy(), preds=y_test_pred.numpy(), class_names=labels
    # )
    # wandb.log({"confusion_matrix": cm, 'outer_fold': outer_fold})

    # from wandb.sklearn import plot_confusion_matrix
    # y_probas = best_model.predict_proba(x_test)
    # plot_roc(y_test, y_probas, labels)
    # plot_precision_recall(y_test, y_probas, labels)

    # plot_learning_curve(model, X_train, y_train)
    # plot_feature_importances(model)

    # VISUALIZATIONS
    # for panel, data in [("train", overall_train), ("test", overall_test)]:
    #     fig, ax = plt.subplots()
    #     pdat = pd.DataFrame.from_records(data)
    #     pdat = pdat.melt(id_vars="outer_fold")
    #     pdat["value"] = pdat.value.astype(float)
    #     sns.boxplot(data=pdat, x="variable", y="value", ax=ax)
    #     sns.stripplot(data=pdat, x="variable", y="value", hue="outer_fold", ax=ax)
    #     ax.set_title(panel)
    #
    #     wandb.log({f"scores/{panel}": wandb.Image(fig)})
    #     plt.close(fig)

    wandb.finish()


# main()

if __name__ == "__main__":
    from jsonargparse import auto_cli

    auto_cli(main, as_positional=False, fail_untyped=False)

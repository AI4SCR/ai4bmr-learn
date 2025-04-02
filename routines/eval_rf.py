# %%
from dataclasses import asdict, dataclass, field
from typing import Any
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import wandb
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from torchmetrics import Accuracy, F1Score, MetricCollection, Precision, Recall

from ai4bmr_learn.data_models.WandInitConfig import WandbInitConfig


@dataclass
class Parameters:
    n_estimators: list[int] = field(default_factory=lambda: [50, 100])
    max_depth: list[int | None] = field(default_factory=lambda: [None])
    min_samples_split: list[int] = field(default_factory=lambda: [2])
    max_features: list[str] = field(default_factory=lambda: ["sqrt"])
    criterion: list[str] = field(default_factory=lambda: ["gini"])


@dataclass
class SweepConfig:
    method: str = "grid"
    num_outer_cv: int = 3
    num_inner_cv: int = 3
    parameters: Parameters = field(default_factory=lambda: Parameters())


# from ai4bmr_learn.datamodules.DummyTabular import DummyDataModule
# dm = datamodule = DummyDataModule()

# %%


def main(
    sweep: SweepConfig,
    datamodule: Any,
    model: Any = None,
    wandb_init: WandbInitConfig = WandbInitConfig(),
):

    # DATA
    dm = datamodule
    dm.prepare_data()
    dm.setup()

    # COMBINE PRE-DEFINED TRAIN AND TEST
    train = [i for i in datamodule.train_set]
    x_train = np.stack([i["x"] for i in train])
    y_train = np.stack([i["target"] for i in train])

    test = [i for i in datamodule.test_set]
    x_test = np.stack([i["x"] for i in test])
    y_test = np.stack([i["target"] for i in test])

    x, y = np.concat([x_train, x_test]), np.concat([y_train, y_test])
    labels = dm.train_set.dataset.labels
    num_classes = len(set(y))

    # METRICS
    task = "multiclass" if num_classes > 2 else "binary"
    metrics_train = MetricCollection(
        {
            "accuracy": Accuracy(task=task, num_classes=num_classes),
            "recall": Recall(task=task, num_classes=num_classes),
            "precision": Precision(task=task, num_classes=num_classes),
            "f1": F1Score(task=task, num_classes=num_classes),
        },
        prefix="train/",
    )
    metrics_test = metrics_train.clone(prefix="test/")

    overall_train = []
    overall_test = []

    # MODEL
    model = RandomForestClassifier(random_state=42) if model is None else model

    # SWEEP
    param_grid = asdict(sweep.parameters)
    outer_cv = StratifiedKFold(n_splits=sweep.num_outer_cv, shuffle=True, random_state=42)
    inner_cv = StratifiedKFold(n_splits=sweep.num_inner_cv, shuffle=True, random_state=42)

    # RUN CONFIGURATION
    wandb.init(**asdict(wandb_init), config=asdict(sweep))

    for outer_fold, (train_idx, test_idx) in enumerate(outer_cv.split(x, y)):
        x_train, x_test = x[train_idx], x[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # GRID SEARCH
        grid = GridSearchCV(
            model,
            param_grid=param_grid,
            scoring="accuracy",
            cv=inner_cv,
            n_jobs=-1,
        )
        grid.fit(x_train, y_train)

        # BEST MODEL
        best_model = grid.best_estimator_
        y_test_pred = best_model.predict(x_test)
        y_train_pred = best_model.predict(x_train)

        # TEST SCORES
        y_test_pred = torch.from_numpy(y_test_pred).long()
        y_test = torch.from_numpy(y_test).long()
        scores_test = metrics_test(y_test_pred, y_test)
        scores_test["outer_fold"] = outer_fold
        wandb.log(scores_test)
        overall_test.append(scores_test)

        # TRAIN SCORES
        y_train_pred = torch.from_numpy(y_train_pred).long()
        y_train = torch.from_numpy(y_train).long()
        scores_train = metrics_train(y_train_pred, y_train)
        scores_train["outer_fold"] = outer_fold
        wandb.log(scores_train)
        overall_train.append(scores_train)

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
    for panel, data in [("train", overall_train), ("test", overall_test)]:
        fig, ax = plt.subplots()
        pdat = pd.DataFrame.from_records(data)
        pdat = pdat.melt(id_vars="outer_fold")
        pdat["value"] = pdat.value.astype(float)
        sns.boxplot(data=pdat, x="variable", y="value", ax=ax)
        sns.stripplot(data=pdat, x="variable", y="value", hue="outer_fold", ax=ax)
        ax.set_title(panel)

        wandb.log({f"scores/{panel}": wandb.Image(fig)})
        plt.close(fig)

    wandb.finish()


# main()

if __name__ == "__main__":
    from jsonargparse import auto_cli

    auto_cli(main, as_positional=False, fail_untyped=False)

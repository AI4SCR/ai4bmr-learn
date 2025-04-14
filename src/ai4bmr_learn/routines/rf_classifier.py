# %%
from dataclasses import asdict, dataclass, field

import pandas as pd
import torch
import wandb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from torchmetrics import MetricCollection

from ai4bmr_learn.data_models.WandInitConfig import WandbInitConfig
from ai4bmr_learn.datamodules.Tabular import TabularDataModule
from ai4bmr_learn.metrics.classification import get_metric_collection


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
    num_outer_cv: int = 5
    num_inner_cv: int = 5
    parameters: Parameters = field(default_factory=lambda: Parameters())


# %%
def rf_classifier(
    datamodule: TabularDataModule,
    sweep: SweepConfig,
    wandb_init: WandbInitConfig = WandbInitConfig(),
    metrics: MetricCollection = None,
):

    # DATA
    dm = datamodule
    dm.prepare_data()
    dm.setup()

    x = dm.dataset.data.values
    y = dm.dataset.targets.values

    labels = dm.train_set.dataset.labels
    num_classes = dm.dataset.num_classes

    # METRICS
    metrics = metrics or get_metric_collection(num_classes=num_classes)
    metrics_train = metrics.clone(prefix="train/")
    metrics_test = metrics_train.clone(prefix="test/")

    overall_train = []
    overall_test = []
    best_params = []

    # MODEL
    model = RandomForestClassifier(random_state=42)

    # SWEEP
    param_grid = asdict(sweep.parameters)
    outer_cv = StratifiedKFold(n_splits=sweep.num_outer_cv, shuffle=True, random_state=42)
    inner_cv = StratifiedKFold(n_splits=sweep.num_inner_cv, shuffle=True, random_state=42)

    # RUN CONFIGURATION
    config = {**asdict(sweep), "target_column_name": dm.target_column_name}
    wandb.init(**asdict(wandb_init), config=config)

    for outer_fold, (train_idx, test_idx) in enumerate(outer_cv.split(x, y)):
        x_train, x_test = x[train_idx], x[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # GRID SEARCH
        grid = GridSearchCV(
            model,
            param_grid=param_grid,
            scoring="balanced_accuracy",
            cv=inner_cv,
            n_jobs=-1,
        )
        grid.fit(x_train, y_train)

        # BEST MODEL
        best_model = grid.best_estimator_
        y_test_pred = best_model.predict(x_test)
        y_train_pred = best_model.predict(x_train)

        # BEST PARAMETERS
        best_params_ = grid.best_params_
        best_params_["outer_fold"] = outer_fold
        best_params.append(best_params_)

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
        from ..logging.class_distribution import log_class_distribution
        from ..logging.confusion_matrix import log_confusion_matrix

        metadata = dict(outer_fold=outer_fold)
        records = [
            dict(
                name=f"train:class_distribution/outer-fold={outer_fold}",
                y_true=y_train,
                y_pred=y_train_pred,
                labels=labels,
            ),
            dict(
                name=f"test:class_distribution/outer-fold={outer_fold}",
                y_true=y_test,
                y_pred=y_test_pred,
                labels=labels,
            ),
        ]
        log_class_distribution(records=records, metadata=metadata)

        # 2. Confusion matrix
        records = [
            dict(
                name=f"train:confusion_matrix/outer-fold={outer_fold}",
                y_true=y_train,
                y_pred=y_train_pred,
                labels=labels,
            ),
            dict(
                name=f"test:confusion_matrix/outer-fold={outer_fold}", y_true=y_test, y_pred=y_test_pred, labels=labels
            ),
        ]
        log_confusion_matrix(records=records, metadata=metadata)

    # VISUALIZATIONS
    from ..logging.log_scores_boxplot import log_scores_boxplot

    records = [
        dict(name="train:scores/", scores=pd.DataFrame(overall_train)),
        dict(name="test:scores/", scores=pd.DataFrame(overall_test)),
    ]
    log_scores_boxplot(records=records)

    best_params = pd.DataFrame(best_params)
    table = wandb.Table(dataframe=best_params)
    wandb.log({"best_params": table})

    wandb.finish()

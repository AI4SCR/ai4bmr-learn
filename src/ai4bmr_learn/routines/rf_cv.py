# %%
from dataclasses import asdict, dataclass, field
import pandas as pd
import torch
import wandb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from torchmetrics import MetricCollection
from datetime import datetime
from ai4bmr_learn.data_models.WandInitConfig import WandbInitConfig
from ai4bmr_learn.datamodules.Tabular import TabularDataModule
from ai4bmr_learn.metrics.classification import get_metric_collection
from sklearn.model_selection import ParameterGrid

param_grid = {
    "n_estimators": [50, 100],
    "max_depth": [None],
    "min_samples_split": [2],
    "max_features": ["sqrt"],
    "criterion": ["gini"],
}

default_params = list(ParameterGrid(param_grid))


@dataclass
class SweepConfig:
    parameters: list[dict] = field(default_factory=lambda: default_params)
    method: str = "grid"
    num_outer_cv: int = 5
    num_inner_cv: int = 5
    test_size: float = 0.2


# %%
def rf_cv(
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
    num_samples = len(dm.dataset)

    # %%
    cv_id = datetime.now().strftime("%Y%m%d-%H:%M:%S")
    save_dir = dm.splits_path.parent / "nested_cv_splits" / cv_id

    # METRICS
    metrics = metrics or get_metric_collection(num_classes=num_classes)
    metrics_train = metrics.clone(prefix="scores:train/")
    metrics_test = metrics_train.clone(prefix="scores:test/")

    # MODEL
    model = RandomForestClassifier(random_state=42)

    # SWEEP
    # note: GridSearch only accepts a list of dicts with parameter lists
    param_grid = [{k: [v] for k, v in i.items()} for i in sweep.parameters]
    outer_cv = StratifiedKFold(n_splits=sweep.num_outer_cv, shuffle=True, random_state=42)
    inner_cv = StratifiedKFold(n_splits=sweep.num_inner_cv, shuffle=True, random_state=42)

    # RUN CONFIGURATION
    metric_for_best_model = "balanced_accuracy"
    for outer_fold, (train_idx, test_idx) in enumerate(outer_cv.split(x, y)):
        config = dict(cv_id=cv_id,
                      cv_type='outer',
                      outer_fold=outer_fold,
                      target_column_name=dm.target_column_name,
                      num_classes=num_classes,
                      num_samples=num_samples,
                      metric_for_best_model=metric_for_best_model)
        wandb_init.config.update(config)
        wandb.init(**asdict(wandb_init))

        x_train, x_test = x[train_idx], x[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # GRID SEARCH
        grid = GridSearchCV(
            model,
            param_grid=param_grid,
            scoring=metric_for_best_model,
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
        best_params_["params_idx"] = int(grid.best_index_)
        wandb.config.update(best_params_)

        cv_results_ = pd.DataFrame.from_dict(grid.cv_results_)
        cv_results_ = cv_results_.drop(columns=['params'])
        cv_results_ = cv_results_.convert_dtypes()
        # NOTE: this is not working due to dtype concersion and other issues at this point
        # table = wandb.Table(dataframe=cv_results_)
        # wandb.log({"cv_results/": table})

        # TEST SCORES
        y_test_pred = torch.from_numpy(y_test_pred).long()
        y_test = torch.from_numpy(y_test).long()
        scores_test = metrics_test(y_test_pred, y_test)
        wandb.log(scores_test)

        # TRAIN SCORES
        y_train_pred = torch.from_numpy(y_train_pred).long()
        y_train = torch.from_numpy(y_train).long()
        scores_train = metrics_train(y_train_pred, y_train)
        wandb.log(scores_train)

        # RESET (not sure if this necessary if we only use the return values of the metrics object)
        metrics_train.reset()
        metrics_test.reset()

        wandb.finish()

        # VISUALIZATIONS
        # 1. Class distribution
        # from ..logging.class_distribution import log_class_distribution
        # from ..logging.confusion_matrix import log_confusion_matrix
        #
        # metadata = dict(outer_fold=outer_fold)
        # records = [
        #     dict(
        #         name=f"train:class_distribution/outer-fold={outer_fold}",
        #         y_true=y_train,
        #         y_pred=y_train_pred,
        #         labels=labels,
        #     ),
        #     dict(
        #         name=f"test:class_distribution/outer-fold={outer_fold}",
        #         y_true=y_test,
        #         y_pred=y_test_pred,
        #         labels=labels,
        #     ),
        # ]
        # log_class_distribution(records=records, metadata=metadata)
        #
        # # 2. Confusion matrix
        # records = [
        #     dict(
        #         name=f"train:confusion_matrix/outer-fold={outer_fold}",
        #         y_true=y_train,
        #         y_pred=y_train_pred,
        #         labels=labels,
        #     ),
        #     dict(
        #         name=f"test:confusion_matrix/outer-fold={outer_fold}", y_true=y_test, y_pred=y_test_pred, labels=labels
        #     ),
        # ]
        # log_confusion_matrix(records=records, metadata=metadata)

    # VISUALIZATIONS
    # from ..logging.log_scores_boxplot import log_scores_boxplot
    #
    # records = [
    #     dict(name="train:scores/", scores=pd.DataFrame(overall_train)),
    #     dict(name="test:scores/", scores=pd.DataFrame(overall_test)),
    # ]
    # log_scores_boxplot(records=records)
    #
    # best_params = pd.DataFrame(best_params)
    # table = wandb.Table(dataframe=best_params)
    # wandb.log({"best_params": table})

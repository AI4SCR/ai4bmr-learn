# %%
from dataclasses import asdict, dataclass, field
import pandas as pd
import torch
import wandb
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from torchmetrics import MetricCollection
from datetime import datetime
from ai4bmr_learn.data_models.WandInitConfig import WandbInitConfig
from ai4bmr_learn.datamodules.TabularLight import TabularDataModule
from ai4bmr_learn.metrics.classification import get_metric_collection
from sklearn.model_selection import ParameterGrid

# %%
param_grid = {
    "C": [0.01, 0.1, 1, 10, 100],
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
def logistic_regression(
        datamodule: TabularDataModule,
        sweep: SweepConfig,
        wandb_init: WandbInitConfig = WandbInitConfig(),
        metrics: MetricCollection = None,
):
    # DATA
    dm = datamodule
    dm.prepare_data()
    dm.setup()

    x = dm.tabular.data.values
    y = dm.tabular.targets.values

    num_classes = dm.tabular.num_classes
    num_samples = dm.tabular.num_samples

    # %%
    cv_id = datetime.now().strftime("%Y%m%d-%H:%M:%S")

    # METRICS
    metrics = metrics or get_metric_collection(num_classes=num_classes)
    metrics_train = metrics.clone(prefix="scores:train/")
    metrics_test = metrics_train.clone(prefix="scores:test/")

    # MODEL
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(max_iter=1000)

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
                      target_column_name=dm.tabular.target_column_name,
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
        # TODO: log tabel to wandb
        cv_results_ = cv_results_.convert_dtypes()

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

# import os
# from ai4bmr_learn.datamodules.DummyTabular import DummyTabular
# os.environ["WANDB_API_KEY"] = ''
#
# dm = DummyTabular()
# wandb_init = WandbInitConfig(name="logistic_regression")
# sweep = SweepConfig()
# logistic_regression(datamodule=dm, sweep=sweep, wandb_init=wandb_init)

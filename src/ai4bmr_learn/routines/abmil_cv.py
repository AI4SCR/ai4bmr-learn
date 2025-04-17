from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd
from sklearn.model_selection import ParameterGrid

from ai4bmr_learn.data_models import WandbInitConfig

# from torchmetrics import MetricCollection
from ai4bmr_learn.datamodules.MIL import MILDataModule
from ai4bmr_learn.datamodules.NestedCV import NestedCV
from ai4bmr_learn.routines.abmil import ABMILConfig, abmil
from ai4bmr_learn.train.train import TrainerConfig

param_grid = {
    "head_dim": [64, 128],
    "n_heads": [1],
    "dropout": [0.0, 0.5],
    "n_branches": [1],
    "gated": [False],
    "hidden_dim": [64],
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
def abmil_cv(
    datamodule: MILDataModule,
    sweep: SweepConfig,
    weighted: bool = False,
    trainer: TrainerConfig = TrainerConfig(),
    wandb_init: WandbInitConfig = WandbInitConfig(),
):

    dm = datamodule
    dm.prepare_data()
    dm.setup()

    cv_id = datetime.now().strftime("%Y%m%d-%H:%M")
    save_dir = dm.splits_path.parent / "nested_cv_splits" / cv_id
    ncv = NestedCV(
        metadata=dm.dataset.metadata,
        save_dir=save_dir,
        test_size=sweep.test_size,
        force=False,
        num_outer_cv=sweep.num_outer_cv,
        num_inner_cv=sweep.num_inner_cv,
    )

    # %% CROSS-VALIDATION
    metric_for_best_model = "scores:test/f1-macro"
    outer_results = pd.DataFrame()
    for outer_fold in range(ncv.num_outer_cv):
        inner_results = []

        for params_idx, params in enumerate(sweep.parameters):
            model_cfg = ABMILConfig(**params)

            for inner_fold in range(ncv.num_inner_cv):
                inner_split_path = ncv.get_dataset_path(outer_fold=outer_fold, inner_fold=inner_fold)
                config = dict(
                    cv_id=cv_id,
                    cv_type="inner",
                    outer_fold=outer_fold,
                    inner_fold=inner_fold,
                    params_idx=params_idx
                )
                wandb_init.config.update(config)

                dm.splits_path = inner_split_path
                results_ = abmil(
                    datamodule=dm,
                    model=model_cfg,
                    trainer=trainer,
                    weighted=weighted,
                    wandb_init=wandb_init,
                    monitor_metric_name="train_loss_epoch",
                )
                results_.update(config)
                inner_results.append(results_)

        # FIND BEST PARAM_IDX
        inner_results = pd.json_normalize(inner_results)
        best_param_idx = (
            inner_results.groupby("params_idx")[[metric_for_best_model, "trainable_params"]]
            .mean()
            .sort_values("trainable_params")
            .index[0]
        )
        model_cfg = ABMILConfig(**sweep.parameters[best_param_idx])

        # EVAL ON TEST WITH BEST PARAMS
        outer_split_path = ncv.get_dataset_path(outer_fold=outer_fold)
        config = dict(cv_id=cv_id, cv_type="outer", outer_fold=outer_fold, inner_fold=None, params_idx=best_param_idx)
        wandb_init.config.update(config)

        dm.splits_path = outer_split_path
        results_ = abmil(
            datamodule=dm,
            model=model_cfg,
            trainer=trainer,
            wandb_init=wandb_init,
            monitor_metric_name="train_loss_epoch",
        )

        results_ = pd.json_normalize(results_)
        outer_results = pd.concat([outer_results, inner_results, results_])

    return outer_results

from datetime import datetime

import pandas as pd
from sklearn.model_selection import ParameterGrid

from ai4bmr_learn.data_models import WandbInitConfig
from ai4bmr_learn.datamodules.DummyMIL import DummyMIL
from ai4bmr_learn.datamodules.NestedCV import NestedCV
from ai4bmr_learn.routines.abmil import ABMILConfig, abmil
from ai4bmr_learn.train.train import TrainerConfig

# %% DATA
dm = DummyMIL()
dm.prepare_data()
dm.setup()

save_dir = dm.splits_path.parent / "nested_cv_splits"
ncv = NestedCV(metadata=dm.dataset.metadata, save_dir=save_dir, test_size=0.2, force=True,
               num_outer_cv=5, num_inner_cv=3)

# %% PARAMETER GRID
param_grid = {
    "head_dim": [64, 128],
    "n_heads": [1],
    "dropout": [0.0],
    "n_branches": [1],
    "gated": [False],
    "hidden_dim": [64],
}
param_grid = list(ParameterGrid(param_grid))

# %% CROSS-VALIDATION
trainer = TrainerConfig(max_epochs=2, log_every_n_steps=50)
timestamp = datetime.now().strftime("%Y%m%d-%H:%M")
metric_for_best_model = "scores:test/f1-macro"
for outer_fold in range(ncv.num_outer_cv):
    results = []

    for params_idx, params in enumerate(param_grid):
        model_cfg = ABMILConfig(**param_grid[params_idx])

        for inner_fold in range(ncv.num_inner_cv):
            inner_split_path = ncv.get_dataset_path(outer_fold=outer_fold, inner_fold=inner_fold)
            config = dict(timestamp=timestamp, cv_type='inner', outer_fold=outer_fold, inner_fold=inner_fold, params_idx=params_idx)
            wandb_init = WandbInitConfig(project='cv', tags=['cv'], config=config)

            dm.splits_path = inner_split_path
            results_ = abmil(datamodule=dm, model=model_cfg, trainer=trainer, wandb_init=wandb_init, monitor_metric_name='train_loss_epoch')
            results_.update(config)
            results.append(results_)

    # FIND BEST PARAM_IDX
    results = pd.json_normalize(results)
    best_param_idx = results \
        .groupby('params_idx')[[metric_for_best_model, 'trainable_params']] \
        .mean() \
        .sort_values('trainable_params') \
        .index[0]
    model_cfg = ABMILConfig(**param_grid[best_param_idx])

    # EVAL ON TEST WITH BEST PARAMS
    outer_split_path = ncv.get_dataset_path(outer_fold=outer_fold)
    config = dict(timestamp=timestamp, cv_type='outer', outer_fold=outer_fold, inner_fold=None, params_idx=best_param_idx)
    wandb_init = WandbInitConfig(project='cv', tags=['cv'], config=config)

    dm.splits_path = outer_split_path
    results_ = abmil(datamodule=dm, model=model_cfg, trainer=trainer, wandb_init=wandb_init, monitor_metric_name='train_loss_epoch')

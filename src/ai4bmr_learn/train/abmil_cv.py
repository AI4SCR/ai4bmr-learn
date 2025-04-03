# %%
from dataclasses import asdict, dataclass, field
from pathlib import Path
from loguru import logger

import pandas as pd
import wandb
from sklearn.model_selection import ParameterGrid

from ai4bmr_learn.data.splits import generate_splits, Split
from ai4bmr_learn.data_models.WandInitConfig import WandbInitConfig
from ai4bmr_learn.datamodules.MIL import MILDataModule
from ai4bmr_learn.logging.log_scores_boxplot import log_scores_boxplot
from ai4bmr_learn.train.abmil import ABMILConfig, abmil
from ai4bmr_learn.train.train import TrainerConfig
from ai4bmr_learn.datamodules.DummyMIL import DummyMIL


@dataclass
class Parameters:
    # TODO: add more tunable parameters
    head_dim: list[int] = field(default_factory=lambda: [16, 32, 64, 128])
    n_heads: list[int] = field(default_factory=lambda: [1, 4])
    hidden_dim: list[int] = field(default_factory=lambda: [32])
    dropout: list[float] = field(default_factory=lambda: [0.0, 0.5])
    gated: list[bool] = field(default_factory=lambda: [False])


@dataclass
class SweepConfig:
    test_size: float = 0.2
    train_size: float = 0.8
    val_size: float = 0.2
    num_outer_cv: int = 2
    num_inner_cv: int = 2
    parameters: Parameters = field(default_factory=lambda: Parameters())


def get_best_param_index(scores: pd.DataFrame, scoring: str, maximize=True):
    # TODO: sort by number of trainable params
    best_idx = scores.groupby(["outer_fold", "parameter_index"])[scoring].mean().sort_values(ascending=maximize)
    best_idx = best_idx.reset_index().parameter_index.iloc[-1]
    return best_idx


def create_fold_dataset(dataset, *, sweep: SweepConfig, save_dir: Path):
    save_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Creating folds in {save_dir}")

    metadata = dataset.metadata
    target_column_name = dataset.target_column_name
    for outer_fold in range(sweep.num_outer_cv):
        outer_metadata = generate_splits(
            metadata, target_column_name=target_column_name, test_size=sweep.test_size, random_state=outer_fold
        )
        outer_metadata.to_parquet(save_dir / f"outer_fold={outer_fold}.parquet", engine="fastparquet")

        for inner_fold in range(sweep.num_inner_cv):
            filter_ = outer_metadata[Split.COLUMN_NAME] == Split.TRAIN
            inner_metadata = outer_metadata[filter_]
            inner_metadata = generate_splits(
                inner_metadata, target_column_name=target_column_name, test_size=sweep.val_size, random_state=inner_fold
            )

            inner_metadata.to_parquet(
                save_dir / f"outer_fold={outer_fold}-inner_fold={inner_fold}.parquet", engine="fastparquet"
            )


class MILCVDataModule(MILDataModule):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _prepare_data(self) -> None:
        pass


# %%
def abmil_cv(
    datamodule: MILDataModule = DummyMIL(),
    sweep: SweepConfig = SweepConfig(),
    trainer: TrainerConfig = TrainerConfig(max_epochs=2),
    wandb_init: WandbInitConfig = WandbInitConfig(),
):
    # DATA
    datamodule.prepare_data()
    datamodule.setup()

    # SPLITS
    splits_dir = datamodule.splits_path.parent / "splits"
    create_fold_dataset(dataset=datamodule.dataset, sweep=sweep, save_dir=splits_dir)

    # CONFIGURE WAND
    wandb.init(**asdict(wandb_init), config=None)

    monitor_metric_name = "train/accuracy-macro"
    scores = pd.DataFrame()
    overall_train = []
    overall_test = []
    param_grid = list(ParameterGrid(param_grid=asdict(sweep.parameters)))
    for outer_fold in range(sweep.num_outer_cv):

        for param_idx, params in enumerate(param_grid):
            model = ABMILConfig(**params)

            for inner_fold in range(sweep.num_inner_cv):
                inner_scores = dict(
                    parameter_index=param_idx, inner_fold=inner_fold, outer_fold=outer_fold, params=params
                )

                splits_path = splits_dir / f"outer_fold={outer_fold}-inner_fold={inner_fold}.parquet"
                assert splits_path.exists()

                # TODO: this does not respect the configuration used to create the input datamodule
                dm = MILCVDataModule(
                    data_dir=datamodule.data_dir,
                    metadata_path=datamodule.metadata_path,
                    splits_path=splits_path,
                    target_column_name=datamodule.target_column_name,
                )

                metadata = dict(inner_fold=inner_fold, outer_fold=outer_fold)
                train_scores, test_scores, model_stats = abmil(
                    datamodule=dm,
                    model=model,
                    weighted=False,
                    monitor_metric_name=monitor_metric_name,  # note: we do not have a val set
                    trainer=trainer,
                    metadata=metadata,
                    metric_identifier="inner_cv",
                    viz_collection=None,
                )

                inner_scores.update(model_stats)
                inner_scores.update(train_scores)
                inner_scores.update(test_scores)
                data = pd.DataFrame.from_dict(inner_scores, orient="index").T
                scores = pd.concat([scores, data])

        best_idx = get_best_param_index(scores, scoring="inner_cv:test/accuracy-macro", maximize=True)
        params = param_grid[best_idx]
        model = ABMILConfig(**params)

        splits_path = splits_dir / f"outer_fold={outer_fold}.parquet"
        dm = MILCVDataModule(
            data_dir=datamodule.data_dir,
            metadata_path=datamodule.metadata_path,
            splits_path=splits_path,
            target_column_name=datamodule.target_column_name,
        )

        metadata = dict(outer_fold=outer_fold)
        train_scores, test_scores, model_stats = abmil(
            datamodule=dm,
            model=model,
            weighted=False,
            monitor_metric_name=monitor_metric_name,
            trainer=trainer,
            metadata=metadata,
            metric_identifier="outer_cv",
        )
        train_scores.update(dict(outer_fold=outer_fold))
        test_scores.update(dict(outer_fold=outer_fold))

        overall_train.append(train_scores)
        overall_test.append(test_scores)

    # %%
    records = [
        dict(name="train", scores=pd.DataFrame(overall_train)),
        dict(name="test", scores=pd.DataFrame(overall_test)),
    ]
    log_scores_boxplot(records=records)

    # log scores
    table = wandb.Table(dataframe=scores)
    wandb.log({"scores": table})

    wandb.finish()

    return scores

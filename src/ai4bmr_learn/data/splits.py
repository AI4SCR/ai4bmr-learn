from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, KFold, GroupKFold, StratifiedShuffleSplit
from torch.utils.data import Dataset


class Split(str, Enum):
    COLUMN_NAME = "split"
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


def generate_splits(
        metadata: pd.DataFrame,
        target_column_name: str = "target",
        test_size: float | None = None,
        val_size: float | None = None,
        stratify: bool = False,
        group_column_name: str | None = None,
        random_state: int | None = None,
        verbose: int = 1,
):
    assert test_size or val_size, "Either `test_size` or `val_size` must be provided"
    num_test_splits = round(1 / test_size) if test_size is not None else None
    num_val_splits = round(1 / val_size) if val_size is not None else None

    # TODO: support Grouped Split for multi-samples per patient cases
    # TODO: should we allow for re-splitting, i.e. check if `split_column_name` already exists in metadata?
    # TODO: should we overwrite the metadata?

    metadata = metadata.copy()
    assert metadata.index.has_duplicates == False
    num_samples = len(metadata)

    if group_column_name is not None:
        groups = metadata[group_column_name].values
    else:
        groups = None

    if stratify and group_column_name is not None:
        splitter = StratifiedGroupKFold
    elif stratify and group_column_name is None:
        splitter = StratifiedKFold
    elif not stratify and group_column_name is not None:
        splitter = GroupKFold
    else:
        splitter = KFold

    # split into train, test
    if test_size:
        split = splitter(n_splits=num_test_splits, shuffle=True, random_state=random_state)

        y = metadata[target_column_name] if stratify else None
        train_indices, test_indices = next(split.split(np.zeros(num_samples), y=y, groups=groups))

        train_indices = metadata.index[train_indices]
        test_indices = metadata.index[test_indices]
    else:
        train_indices = metadata.index
        test_indices = []

    # split train into train, val
    if val_size:
        train_metadata, test_metadata = (
            metadata.loc[train_indices],
            metadata.loc[test_indices],
        )
        num_train_samples = len(train_metadata)

        split = splitter(n_splits=num_val_splits, shuffle=True, random_state=random_state)

        y = train_metadata[target_column_name] if stratify else None
        train_indices, val_indices = next(split.split(np.zeros(num_train_samples), y=y, groups=groups))

        train_indices = train_metadata.index[train_indices]
        val_indices = train_metadata.index[val_indices]
    else:
        val_indices = []

    # sanity checks
    assert set(train_indices).union(val_indices).union(test_indices) == set(metadata.index.values)
    assert set(train_indices).intersection(test_indices) == set()
    assert set(val_indices).intersection(test_indices) == set()

    logger.info(
        f"Split dataset in:\n"
        f"  num_test: {len(test_indices)}\n"
        f"  num_train: {len(train_indices)}\n"
        f"  num_val: {len(val_indices)}"
    )

    # NOTE: we need to use the `.value` otherwise the column names is `Split.COLUM_NAME` after re-load
    metadata.loc[test_indices, Split.COLUMN_NAME.value] = Split.TEST.value
    metadata.loc[train_indices, Split.COLUMN_NAME.value] = Split.TRAIN.value
    metadata.loc[val_indices, Split.COLUMN_NAME.value] = Split.VAL.value
    assert metadata[Split.COLUMN_NAME.value].isna().any() == False
    dtype = pd.CategoricalDtype(categories=[Split.TRAIN.value, Split.VAL.value, Split.TEST.value], ordered=False)
    metadata[Split.COLUMN_NAME.value] = metadata[Split.COLUMN_NAME.value].astype(dtype)

    return metadata


def create_nested_cv_datasets(
        *,
        metadata: pd.DataFrame,
        num_outer_cv: int = 5,
        num_inner_cv: int = 5,
        target_column_name: str = "target",
        test_size: float | None = None,
        val_size: float | None = None,
        stratify: bool = False,
        save_dir: Path,
):
    save_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Creating folds in {save_dir}")

    for outer_fold in range(num_outer_cv):
        outer_metadata = generate_splits(
            metadata,
            target_column_name=target_column_name,
            test_size=test_size,
            stratify=stratify,
            random_state=outer_fold,
        )
        outer_metadata.to_parquet(save_dir / f"outer_fold={outer_fold}.parquet", engine="fastparquet")

        for inner_fold in range(num_inner_cv):
            filter_ = outer_metadata[Split.COLUMN_NAME] == Split.TRAIN
            inner_metadata = outer_metadata[filter_]
            inner_metadata = generate_splits(
                inner_metadata,
                target_column_name=target_column_name,
                test_size=test_size,
                val_size=val_size,
                random_state=inner_fold,
            )

            inner_metadata.to_parquet(
                save_dir / f"outer_fold={outer_fold}-inner_fold={inner_fold}.parquet", engine="fastparquet"
            )


def has_splits(metadata: pd.DataFrame):
    return Split.COLUMN_NAME.value in metadata.columns


def generate_subset(dataset: Dataset, metadata: pd.DataFrame | None, num_samples: int | None = None):
    from torch.utils.data import Subset

    subset_idx = np.random.choice(len(metadata), num_samples, replace=False)
    return Subset(dataset, subset_idx), metadata.iloc[subset_idx]

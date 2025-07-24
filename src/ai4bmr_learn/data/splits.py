from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, KFold, GroupKFold, StratifiedShuffleSplit
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder


class Split(str, Enum):
    COLUMN_NAME = "split"
    TRAIN = "train"
    FIT = "fit"
    VAL = "val"
    TEST = "test"


def generate_splits(
        metadata: pd.DataFrame,
        test_size: float | None = None,
        val_size: float | None = None,
        stratify: bool = False,
        target_column_name: str | None = None,
        encode_targets: bool = False,
        nan_value: int = -1,
        use_filtered_targets_for_train: bool = False,
        include_targets: list[str] | None = None,
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
    indices_universe = metadata.index.values

    # FILTER DATA
    if target_column_name is not None:
        assert target_column_name in metadata
        targets = metadata[target_column_name]
        filter_ = targets.notna().values
        targets = targets[filter_]

        if include_targets is not None:
            filter_ = targets.isin(include_targets)
            targets = targets[filter_]

        indices = targets.index.values
    else:
        indices = indices_universe

    num_samples = len(indices)

    if encode_targets:
        mapping = {v:k for k, v in enumerate(targets.unique())}
        metadata.loc[:, target_column_name] = metadata[target_column_name].transform(lambda x: mapping.get(x, nan_value))
        metadata = metadata.astype({target_column_name: int})

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
        y = np.array(['a'] * 5 + ['b'] * 5).astype('object')
        next(split.split(np.zeros(10), y=y, groups=None))
        y = metadata.loc[indices, target_column_name] if stratify else None
        groups = metadata.loc[indices, group_column_name].values if group_column_name is not None else None
        train_indices, test_indices = next(split.split(np.zeros(num_samples), y=y, groups=groups))

        train_indices = indices[train_indices]
        test_indices = indices[test_indices]
    else:
        train_indices = indices
        test_indices = []

    # split train into fit, val
    if val_size:
        train_metadata, test_metadata = metadata.loc[train_indices], metadata.loc[test_indices]
        num_train_samples = len(train_metadata)

        split = splitter(n_splits=num_val_splits, shuffle=True, random_state=random_state)

        y = train_metadata[target_column_name] if stratify else None
        groups = train_metadata[group_column_name].values if group_column_name is not None else None
        fit_indices, val_indices = next(split.split(np.zeros(num_train_samples), y=y, groups=groups))

        fit_indices = train_metadata.index[fit_indices]
        val_indices = train_metadata.index[val_indices]
    else:
        val_indices = []
        fit_indices = train_indices

    if use_filtered_targets_for_train:
        excl_indices = set(indices_universe) - set(fit_indices).union(val_indices).union(test_indices)
        fit_indices = list(set(fit_indices).union(excl_indices))
        assert set(fit_indices).union(val_indices).union(test_indices) == set(indices_universe)
    else:
        assert set(fit_indices).union(val_indices).union(test_indices) == set(indices)

    # sanity check
    assert set(train_indices).intersection(test_indices) == set()
    assert set(val_indices).intersection(test_indices) == set()
    assert set(fit_indices).intersection(test_indices) == set()
    assert set(fit_indices).intersection(val_indices) == set()

    print_split_summary(metadata=metadata, fit_indices=fit_indices, test_indices=test_indices, val_indices=val_indices)

    # NOTE: we need to use the `.value` otherwise the column names is `Split.COLUM_NAME` after re-load
    metadata.loc[test_indices, Split.COLUMN_NAME.value] = Split.TEST.value
    metadata.loc[fit_indices, Split.COLUMN_NAME.value] = Split.FIT.value
    metadata.loc[val_indices, Split.COLUMN_NAME.value] = Split.VAL.value

    if use_filtered_targets_for_train:
        assert metadata[Split.COLUMN_NAME.value].isna().any() == False

    dtype = pd.CategoricalDtype(categories=[Split.FIT.value, Split.VAL.value, Split.TEST.value], ordered=False)
    metadata[Split.COLUMN_NAME.value] = metadata[Split.COLUMN_NAME.value].astype(dtype)

    return metadata


from rich.console import Console
from rich.table import Table

def print_split_summary(metadata, test_indices, fit_indices, val_indices):
    all_indices = set(fit_indices).union(val_indices).union(test_indices)

    console = Console()
    table = Table(title=f"Dataset Split (n={len(metadata)})")

    # define columns
    table.add_column("Split", justify="left", style="bold")
    table.add_column("Count", justify="right")

    # add rows
    table.add_row("Test", str(len(test_indices)))
    table.add_row("Fit",  str(len(fit_indices)))
    table.add_row("Val",  str(len(val_indices)))
    table.add_row("───",  "─" * max(len(str(len(all_indices))), 3))
    table.add_row("Total", str(len(all_indices)))

    console.print(table)

# Example usage:
# print_split_summary(metadata, test_indices, fit_indices, val_indices, all_indices)

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
            filter_ = outer_metadata[Split.COLUMN_NAME] == Split.FIT
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

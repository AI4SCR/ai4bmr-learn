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
    """
    Generates a single train/test/validation split for a given metadata DataFrame.

    Args:
        metadata: The input DataFrame containing metadata.
        test_size: The proportion of the dataset to include in the test split.
                   Used to calculate n_splits for KFold as round(1 / test_size).
                   The actual test set proportion will be 1 / round(1 / test_size).
        val_size: The proportion of the training dataset to include in the validation split.
                  Used to calculate n_splits for KFold as round(1 / val_size).
                  The actual validation set proportion will be 1 / round(1 / val_size).
        stratify: If True, data is split in a stratified fashion.
        target_column_name: The name of the column to use for stratification.
        encode_targets: If True, encodes the target column to integers.
        nan_value: Value to use for NaN targets when encoding.
        use_filtered_targets_for_train: If True, samples that were initially filtered out (e.g., due to missing
                                        `target_column_name` values or not being in `include_targets`)
                                        are added back to the 'fit' split. This means the 'fit' split
                                        might contain samples without valid targets, and stratification
                                        will not apply to these added samples. Use with caution.
        include_targets: A list of target values to include in the splitting process.
        group_column_name: The name of the column to use for grouping.
        random_state: Seed for random number generator for reproducibility.
        verbose: Verbosity level.

    Returns:
        A copy of the metadata DataFrame with an added 'split' column indicating 'fit', 'val', or 'test'.
    """
    assert test_size or val_size, "Either `test_size` or `val_size` must be provided"
    num_test_splits = round(1 / test_size) if test_size is not None else None
    num_val_splits = round(1 / val_size) if val_size is not None else None

    # TODO: should we allow for re-splitting, i.e. check if `split_column_name` already exists in metadata?

    metadata = metadata.copy()
    assert metadata.index.has_duplicates == False, f'Index of metadata needs to be unique'
    indices_universe = metadata.index.values
    index_names = metadata.index.names

    # metadata = metadata.reset_index()
    # metadata.set_index(index_names, inplace=True)

    # FILTER DATA
    if target_column_name is not None:
        assert target_column_name in metadata, f'{target_column_name} not in metadata columns'
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
        mapping = {v: k for k, v in enumerate(sorted(targets.unique()))}
        metadata[target_column_name] = metadata[target_column_name].tolist()  # note: get rid of category if any
        metadata.loc[:, target_column_name] = metadata[target_column_name].transform(
            lambda x: mapping.get(x, nan_value))
        metadata[target_column_name] = metadata[target_column_name].fillna(nan_value)
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
        assert set(fit_indices).union(val_indices).union(test_indices) == set(indices_universe), f'Index sets do not match'
    else:
        assert set(fit_indices).union(val_indices).union(test_indices) == set(indices), f'Index sets do not match'

    # sanity check
    assert set(train_indices).intersection(test_indices) == set(), f'Overlap between train and test indices'
    assert set(val_indices).intersection(test_indices) == set(), f'Overlap between val and test indices'
    assert set(fit_indices).intersection(test_indices) == set(), f'Overlap between fit and test indices'
    assert set(fit_indices).intersection(val_indices) == set(), f'Overlap between fit and val indices'

    print_split_summary(metadata=metadata, fit_indices=fit_indices, test_indices=test_indices, val_indices=val_indices)

    # NOTE: we need to use the `.value` otherwise the column names is `Split.COLUM_NAME` after re-load
    metadata.loc[test_indices, Split.COLUMN_NAME.value] = Split.TEST.value
    metadata.loc[fit_indices, Split.COLUMN_NAME.value] = Split.FIT.value
    metadata.loc[val_indices, Split.COLUMN_NAME.value] = Split.VAL.value

    if use_filtered_targets_for_train:
        assert metadata[Split.COLUMN_NAME.value].isna().any() == False, f'Some samples do not have a split assigned'

    dtype = pd.CategoricalDtype(categories=[Split.FIT.value, Split.VAL.value, Split.TEST.value], ordered=False)
    metadata[Split.COLUMN_NAME.value] = metadata[Split.COLUMN_NAME.value].astype(dtype)

    return metadata


def save_splits(
        metadata: pd.DataFrame,
        save_dir: Path,
        test_size: float,
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
        overwrite: bool = False,
):
    """
    Generates a single train/test/validation split for a given metadata DataFrame.

    Args:
        metadata: The input DataFrame containing metadata.
        save_dir: Path under which the splits are saved.
        test_size: The proportion of the dataset to include in the test split.
                   Used to calculate n_splits for KFold as round(1 / test_size).
                   The actual test set proportion will be 1 / round(1 / test_size).
        val_size: The proportion of the training dataset to include in the validation split.
                  Used to calculate n_splits for KFold as round(1 / val_size).
                  The actual validation set proportion will be 1 / round(1 / val_size).
        stratify: If True, data is split in a stratified fashion.
        target_column_name: The name of the column to use for stratification.
        encode_targets: If True, encodes the target column to integers.
        nan_value: Value to use for NaN targets when encoding.
        use_filtered_targets_for_train: If True, samples that were initially filtered out (e.g., due to missing
                                        `target_column_name` values or not being in `include_targets`)
                                        are added back to the 'fit' split. This means the 'fit' split
                                        might contain samples without valid targets, and stratification
                                        will not apply to these added samples. Use with caution.
        include_targets: A list of target values to include in the splitting process.
        group_column_name: The name of the column to use for grouping.
        random_state: Seed for random number generator for reproducibility.
        verbose: Verbosity level.

    Returns:
        None
    """
    if not overwrite:
        assert not save_dir.exists(), f'{save_dir} already exists.'

    num_test_splits = round(1 / test_size)
    num_val_splits = round(1 / val_size) if val_size is not None else None

    metadata = metadata.copy()
    assert metadata.index.is_unique, f'Index of metadata needs to be unique'
    universe = metadata.index.values
    index_names = metadata.index.names

    # FILTER DATA
    if target_column_name is not None:
        assert target_column_name in metadata, f'{target_column_name} not in metadata columns'
        targets = metadata[target_column_name]
        filter_ = targets.notna().values
        targets = targets[filter_]

        if include_targets is not None:
            filter_ = targets.isin(include_targets)
            targets = targets[filter_]

        indices = targets.index.values
    else:
        indices = universe

    if encode_targets:
        mapping = {v: k for k, v in enumerate(sorted(targets.unique()))}
        metadata[target_column_name] = metadata[target_column_name].tolist()  # note: get rid of category if any
        metadata.loc[:, target_column_name] = metadata[target_column_name].transform(
            lambda x: mapping.get(x, nan_value))
        metadata[target_column_name] = metadata[target_column_name].fillna(nan_value)
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
    train_test_split = splitter(n_splits=num_test_splits, shuffle=True, random_state=random_state)
    num_samples = len(indices)
    y = metadata.loc[indices, target_column_name] if stratify else None
    groups = metadata.loc[indices, group_column_name].values if group_column_name is not None else None
    for outer, (train_indices, test_indices) in enumerate(
            train_test_split.split(np.zeros(num_samples), y=y, groups=groups)):

        train_indices = indices[train_indices]
        test_indices = indices[test_indices]

        # split train into fit, val
        if val_size:
            fit_val_split = splitter(n_splits=num_val_splits, shuffle=True, random_state=random_state)
            train_metadata, test_metadata = metadata.loc[train_indices], metadata.loc[test_indices]
            num_train_samples = len(train_metadata)

            y = train_metadata[target_column_name] if stratify else None
            groups = train_metadata[group_column_name].values if group_column_name is not None else None
            for inner, (fit_indices, val_indices) in enumerate(
                    fit_val_split.split(np.zeros(num_train_samples), y=y, groups=groups)):

                fit_indices = train_metadata.index[fit_indices]
                val_indices = train_metadata.index[val_indices]

                metadata = construct_split(metadata=metadata.copy(),
                                           universe=universe, indices=indices,
                                           test_indices=test_indices, train_indices=train_indices,
                                           fit_indices=fit_indices, val_indices=val_indices,
                                           use_filtered_targets_for_train=use_filtered_targets_for_train)
                save_dir.mkdir(parents=True, exist_ok=True)
                metadata.to_parquet(save_dir / f'outer={outer}-inner={inner}-seed={random_state}.parquet', engine='fastparquet')

        val_indices = []
        fit_indices = train_indices

        metadata = construct_split(metadata=metadata.copy(), universe=universe, indices=indices,
                                   test_indices=test_indices, train_indices=train_indices,
                                   fit_indices=fit_indices, val_indices=val_indices,
                                   use_filtered_targets_for_train=use_filtered_targets_for_train)
        save_dir.mkdir(parents=True, exist_ok=True)
        metadata.to_parquet(save_dir / f'outer={outer}-seed={random_state}.parquet', engine='fastparquet')


def construct_split(*, metadata, universe, indices,
                    test_indices, train_indices, fit_indices, val_indices,
                    use_filtered_targets_for_train):

    if use_filtered_targets_for_train:
        excl_indices = set(universe) - set(fit_indices).union(val_indices).union(test_indices)
        fit_indices = list(set(fit_indices).union(excl_indices))
        assert set(fit_indices).union(val_indices).union(test_indices) == set(universe)
    else:
        assert set(fit_indices).union(val_indices).union(test_indices) == set(indices)

    # sanity checks
    assert set(train_indices).intersection(test_indices) == set(), f'Overlap between train and test indices'
    assert set(val_indices).intersection(test_indices) == set(), f'Overlap between val and test indices'
    assert set(fit_indices).intersection(test_indices) == set(), f'Overlap between fit and test indices'
    assert set(fit_indices).intersection(val_indices) == set(), f'Overlap between fit and val indices'
    print_split_summary(metadata=metadata,
                        fit_indices=fit_indices, test_indices=test_indices, val_indices=val_indices)
    # NOTE: we need to use the `.value` otherwise the column names is `Split.COLUM_NAME` after re-load
    metadata.loc[test_indices, Split.COLUMN_NAME.value] = Split.TEST.value
    metadata.loc[fit_indices, Split.COLUMN_NAME.value] = Split.FIT.value
    metadata.loc[val_indices, Split.COLUMN_NAME.value] = Split.VAL.value
    if use_filtered_targets_for_train:
        assert metadata[Split.COLUMN_NAME.value].isna().any() == False, f'Some samples do not have a split assigned'
    dtype = pd.CategoricalDtype(categories=[Split.FIT.value, Split.VAL.value, Split.TEST.value], ordered=False)
    metadata[Split.COLUMN_NAME.value] = metadata[Split.COLUMN_NAME.value].astype(dtype)
    return metadata


from rich.console import Console
from rich.table import Table


def print_split_summary(metadata, test_indices, fit_indices, val_indices):
    all_indices = set(fit_indices).union(val_indices).union(test_indices)
    total_samples = len(metadata)

    console = Console()
    table = Table(title=f"Dataset Split (n={total_samples})")

    # define columns
    table.add_column("Split", justify="left", style="bold")
    table.add_column("Count", justify="right")

    # add rows
    table.add_row("Test", f"{len(test_indices)} ({len(test_indices) / total_samples:.1%})")
    table.add_row("Fit", f"{len(fit_indices)} ({len(fit_indices) / total_samples:.1%})")
    table.add_row("Val", f"{len(val_indices)} ({len(val_indices) / total_samples:.1%})")
    table.add_row("───", "─" * max(len(str(len(all_indices))), 3))
    table.add_row("Total", f"{len(all_indices)} ({len(all_indices) / total_samples:.1%})")

    console.print(table)


def has_splits(metadata: pd.DataFrame):
    return Split.COLUMN_NAME.value in metadata.columns


def generate_subset(dataset: Dataset, metadata: pd.DataFrame | None, num_samples: int | None = None):
    from torch.utils.data import Subset

    subset_idx = np.random.choice(len(metadata), num_samples, replace=False)
    return Subset(dataset, subset_idx), metadata.iloc[subset_idx]

from enum import Enum

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.model_selection import GroupKFold, KFold, StratifiedGroupKFold, StratifiedKFold
from torch.utils.data import Dataset, Subset


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
) -> pd.DataFrame:
    assert test_size or val_size, "Either test_size or val_size must be provided"
    assert metadata.index.is_unique, "Metadata index must be unique"

    metadata = metadata.copy()
    indices_universe = metadata.index.values
    indices = _get_valid_indices(
        metadata=metadata,
        target_column_name=target_column_name,
        include_targets=include_targets,
    )

    if encode_targets:
        assert target_column_name is not None, "encode_targets requires target_column_name"
        _encode_targets(metadata=metadata, target_column_name=target_column_name, nan_value=nan_value)

    train_indices, test_indices = _split_once(
        metadata=metadata,
        indices=indices,
        split_size=test_size,
        stratify=stratify,
        target_column_name=target_column_name,
        group_column_name=group_column_name,
        random_state=random_state,
    )

    fit_indices, val_indices = _split_once(
        metadata=metadata,
        indices=train_indices,
        split_size=val_size,
        stratify=stratify,
        target_column_name=target_column_name,
        group_column_name=group_column_name,
        random_state=random_state,
    )

    if val_size is None:
        fit_indices = train_indices
        val_indices = []

    return _construct_split(
        metadata=metadata,
        universe=indices_universe,
        indices=indices,
        fit_indices=fit_indices,
        val_indices=val_indices,
        test_indices=test_indices,
        use_filtered_targets_for_train=use_filtered_targets_for_train,
    )


def has_splits(metadata: pd.DataFrame) -> bool:
    return Split.COLUMN_NAME.value in metadata.columns


def generate_subset(dataset: Dataset, metadata: pd.DataFrame | None, num_samples: int | None = None):
    assert metadata is not None, "metadata is required"
    assert num_samples is not None, "num_samples is required"
    assert 0 < num_samples <= len(metadata), "num_samples must be in [1, len(metadata)]"

    subset_idx = np.random.choice(len(metadata), num_samples, replace=False)
    return Subset(dataset, subset_idx), metadata.iloc[subset_idx]


def _get_valid_indices(
        *,
        metadata: pd.DataFrame,
        target_column_name: str | None,
        include_targets: list[str] | None,
):
    if target_column_name is None:
        return metadata.index.values

    assert target_column_name in metadata.columns, f"{target_column_name} not in metadata"
    targets = metadata[target_column_name]
    targets = targets[targets.notna()]
    if include_targets is not None:
        targets = targets[targets.isin(include_targets)]
    return targets.index.values


def _encode_targets(*, metadata: pd.DataFrame, target_column_name: str, nan_value: int) -> None:
    unique_targets = sorted(metadata[target_column_name].dropna().unique())
    mapping = {value: index for index, value in enumerate(unique_targets)}
    metadata[target_column_name] = metadata[target_column_name].tolist()
    metadata.loc[:, target_column_name] = metadata[target_column_name].map(lambda x: mapping.get(x, nan_value))
    metadata[target_column_name] = metadata[target_column_name].fillna(nan_value).astype(int)


def _split_once(
        *,
        metadata: pd.DataFrame,
        indices,
        split_size: float | None,
        stratify: bool,
        target_column_name: str | None,
        group_column_name: str | None,
        random_state: int | None,
):
    if split_size is None:
        return indices, []

    num_splits = round(1 / split_size)
    splitter_cls = _get_splitter_cls(stratify=stratify, group_column_name=group_column_name)
    splitter = splitter_cls(n_splits=num_splits, shuffle=True, random_state=random_state)

    y = metadata.loc[indices, target_column_name] if stratify else None
    groups = metadata.loc[indices, group_column_name].values if group_column_name is not None else None
    train_idx, holdout_idx = next(splitter.split(np.zeros(len(indices)), y=y, groups=groups))
    return indices[train_idx], indices[holdout_idx]


def _get_splitter_cls(*, stratify: bool, group_column_name: str | None):
    if stratify and group_column_name is not None:
        return StratifiedGroupKFold
    if stratify:
        return StratifiedKFold
    if group_column_name is not None:
        return GroupKFold
    return KFold


def _construct_split(
        *,
        metadata: pd.DataFrame,
        universe,
        indices,
        fit_indices,
        val_indices,
        test_indices,
        use_filtered_targets_for_train: bool,
) -> pd.DataFrame:
    if use_filtered_targets_for_train:
        excluded_indices = set(universe) - set(fit_indices).union(val_indices).union(test_indices)
        fit_indices = list(set(fit_indices).union(excluded_indices))
        assert set(fit_indices).union(val_indices).union(test_indices) == set(universe)
    else:
        assert set(fit_indices).union(val_indices).union(test_indices) == set(indices)

    assert set(fit_indices).isdisjoint(val_indices), "Overlap between fit and val indices"
    assert set(fit_indices).isdisjoint(test_indices), "Overlap between fit and test indices"
    assert set(val_indices).isdisjoint(test_indices), "Overlap between val and test indices"

    _log_split_summary(metadata=metadata, fit_indices=fit_indices, val_indices=val_indices, test_indices=test_indices)

    metadata.loc[test_indices, Split.COLUMN_NAME.value] = Split.TEST.value
    metadata.loc[fit_indices, Split.COLUMN_NAME.value] = Split.FIT.value
    metadata.loc[val_indices, Split.COLUMN_NAME.value] = Split.VAL.value

    if use_filtered_targets_for_train:
        assert not metadata[Split.COLUMN_NAME.value].isna().any(), "Some samples do not have a split assigned"

    dtype = pd.CategoricalDtype(
        categories=[Split.FIT.value, Split.VAL.value, Split.TEST.value],
        ordered=False,
    )
    metadata[Split.COLUMN_NAME.value] = metadata[Split.COLUMN_NAME.value].astype(dtype)
    return metadata


def _log_split_summary(*, metadata: pd.DataFrame, fit_indices, val_indices, test_indices) -> None:
    total_samples = len(metadata)
    assigned_samples = len(set(fit_indices).union(val_indices).union(test_indices))
    logger.info(
        "Dataset split: fit={} ({:.1%}), val={} ({:.1%}), test={} ({:.1%}), total={} ({:.1%})",
        len(fit_indices),
        len(fit_indices) / total_samples if total_samples else 0.0,
        len(val_indices),
        len(val_indices) / total_samples if total_samples else 0.0,
        len(test_indices),
        len(test_indices) / total_samples if total_samples else 0.0,
        assigned_samples,
        assigned_samples / total_samples if total_samples else 0.0,
    )

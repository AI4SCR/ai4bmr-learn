import pandas as pd
from loguru import logger

from ai4bmr_learn.data.splits import Split


def filter_items_and_metadata(
        item_ids: list[str | int],
        metadata: pd.DataFrame,
        *,
        split: str | None = None,
        drop_nan_columns: bool = False,
):
    """
    Filters a list of item dicts and metadata based on a split value and optionally drops
    NaN-containing columns in the metadata. Returns (items, item_ids, metadata).
    """
    item_ids = set(item_ids)

    # filter item ids
    if split is not None:
        keep = metadata[Split.COLUMN_NAME.value] == split
        assert keep.sum() > 0, f"There are no items that belong to split='{split}'"

        metadata = metadata[keep]
        valid_ids = set(metadata.index)
        logger.info(f"Filtered items and metadata for split={split}.")
    else:
        valid_ids = set(metadata.index)

    if item_ids <= valid_ids:
        logger.warning(f'Not all items have metadata. Dropping items without metadata.')

    valid_item_ids = item_ids.intersection(valid_ids)
    logger.info(f"Found {len(valid_item_ids)} items with metadata.")

    # drop NaN
    filter_ = metadata.isna().any()
    cols_with_nan = metadata.columns[filter_].tolist()
    if filter_.any():
        if drop_nan_columns:
            metadata = metadata.drop(columns=cols_with_nan)

        msg = (
                f"Detected NaN values in the metadata which is incompatible with torch.DataLoader. "
                + ("Dropping them. " if drop_nan_columns else "Use drop_nan_columns=True to drop them. ")
                + f"Affected columns: {cols_with_nan}. "
        )
        logger.warning(msg)

        if split is not None and Split.COLUMN_NAME.value in cols_with_nan:
            logger.warning("Detected NaN in column 'split'. This is most likely a bug.")

    return valid_item_ids, metadata

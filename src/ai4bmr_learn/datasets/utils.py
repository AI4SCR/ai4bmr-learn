import pandas as pd
from loguru import logger


def filter_items_and_metadata(
        item_ids: list[str | int],
        metadata: pd.DataFrame,
        *,
        split: str | None = None,
        drop_nan_columns: bool = False,
):
    item_ids = set(item_ids)

    if split is not None:
        logger.info(f"Filter items and metadata for split={split}.")

        keep = metadata["split"] == split
        assert keep.sum() > 0, f"There are no items that belong to split='{split}'"

        metadata = metadata[keep]
        valid_ids = set(metadata.index)
    else:
        valid_ids = set(metadata.index)

    if not item_ids < valid_ids:
        logger.warning("Not all items have metadata. Dropping items without metadata.")

    valid_item_ids = item_ids.intersection(valid_ids)
    logger.info(f"Found {len(valid_item_ids)}/{len(item_ids)} items with metadata.")

    filter_ = metadata.isna().any()
    cols_with_nan = metadata.columns[filter_].tolist()
    if filter_.any():
        if drop_nan_columns:
            metadata = metadata.drop(columns=cols_with_nan)

        msg = (
                "Detected NaN values in the metadata which is incompatible with torch.DataLoader. "
                + ("Dropping them. " if drop_nan_columns else "Use drop_nan_columns=True to drop them. ")
                + f"Affected columns: {cols_with_nan}. "
        )
        logger.warning(msg)

        if split is not None and "split" in cols_with_nan:
            logger.warning("Detected NaN in column 'split'. This is most likely a bug.")

    return valid_item_ids, metadata.loc[list(valid_item_ids)]

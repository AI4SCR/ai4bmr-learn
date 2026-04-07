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
    ordered_item_ids = list(item_ids)
    metadata_ids = set(metadata.index)
    missing_item_ids = set(ordered_item_ids) - metadata_ids

    if missing_item_ids:
        logger.warning(f"Not all items have metadata. Dropping {len(missing_item_ids)} items without metadata.")

    valid_metadata = metadata
    covered_item_ids = [item_id for item_id in ordered_item_ids if item_id in metadata_ids]
    logger.info(f"Found {len(covered_item_ids)}/{len(ordered_item_ids)} items with metadata.")

    if split is not None:
        logger.info(f"Filter items and metadata for split={split}.")

        keep = valid_metadata[Split.COLUMN_NAME.value] == split
        assert keep.sum() > 0, f"There are no items that belong to split='{split}'"

        valid_metadata = valid_metadata[keep]
        selected_ids = set(valid_metadata.index)
        valid_item_ids = [item_id for item_id in covered_item_ids if item_id in selected_ids]
        logger.info(f"Selected {len(valid_item_ids)}/{len(covered_item_ids)} items for split={split}.")
    else:
        valid_item_ids = covered_item_ids

    valid_metadata = valid_metadata.loc[valid_item_ids]

    filter_ = valid_metadata.isna().any()
    cols_with_nan = valid_metadata.columns[filter_].tolist()
    if filter_.any():
        if drop_nan_columns:
            valid_metadata = valid_metadata.drop(columns=cols_with_nan)

        msg = (
                "Detected NaN values in the metadata which is incompatible with torch.DataLoader. "
                + ("Dropping them. " if drop_nan_columns else "Use drop_nan_columns=True to drop them. ")
                + f"Affected columns: {cols_with_nan}. "
        )
        logger.warning(msg)

        if split is not None and Split.COLUMN_NAME.value in cols_with_nan:
            logger.warning("Detected NaN in column 'split'. This is most likely a bug.")

    return valid_item_ids, valid_metadata

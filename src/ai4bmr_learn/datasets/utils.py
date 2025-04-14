from pathlib import Path

from loguru import logger

from ai4bmr_learn.data.splits import Split, generate_splits


def create_nested_cv_datasets(
    dataset,
    *,
    test_size: float = 0.2,
    val_size: float = 0.2,
    num_outer_cv: int = 5,
    num_inner_cv: int = 5,
    stratify: bool = False,
    save_dir: Path,
):
    save_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Creating folds in {save_dir}")

    metadata = dataset.metadata
    target_column_name = dataset.target_column_name
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
                test_size=val_size,
                val_size=val_size,
                stratify=stratify,
                random_state=inner_fold,
            )

            inner_metadata.to_parquet(
                save_dir / f"outer_fold={outer_fold}-inner_fold={inner_fold}.parquet", engine="fastparquet"
            )

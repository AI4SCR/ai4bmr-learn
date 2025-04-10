from ..data.splits import create_nested_cv_datasets
import pandas as pd
from pathlib import Path


class NestedCV:

    def __init__(
        self,
        *,
        metadata: pd.DataFrame, save_dir: Path, force: bool = False,
        num_outer_cv: int = 5,
        num_inner_cv: int = 5,
        target_column_name: str | None = None,
        test_size: float = 0.2,
        val_size: float | None = None,
        stratify: bool = False,
    ):

        self.num_outer_cv = num_outer_cv
        self.num_inner_cv = num_inner_cv
        self.save_dir = save_dir
        self.target_column_name: str = None,
        self.test_size = test_size
        self.val_size = val_size
        self.target_column_name = target_column_name
        self.metadata = metadata
        self.stratify = stratify

        if save_dir.exists() and not force:
            return
        else:
            create_nested_cv_datasets(
                metadata=metadata,
                num_outer_cv=self.num_outer_cv,
                num_inner_cv=self.num_inner_cv,
                target_column_name=target_column_name,
                test_size=self.test_size,
                val_size=self.val_size,
                stratify=self.stratify,
                save_dir=save_dir,
            )

    def get_dataset_path(self, outer_fold: int, inner_fold: int = None) -> Path:
        path = (
            self.save_dir / f"outer_fold={outer_fold}-inner_fold={inner_fold}.parquet"
            if inner_fold is not None
            else self.save_dir / f"outer_fold={outer_fold}.parquet"
        )
        return path

from ..data.splits import create_nested_cv_datasets
import pandas as pd
from pathlib import Path


class NestedCV:

    def __init__(
        self,
        *,
        metadata: pd.DataFrame,
        save_dir: Path,
        num_outer_cv: int = 5,
        num_inner_cv: int = 5,
        target_column_name: str = "target",
        test_size: float | None = None,
        val_size: float | None = None,
        stratify: bool = False,
        random_state: int = 42,
    ):

        self.num_outer_cv = num_outer_cv
        self.num_inner_cv = num_inner_cv
        self.save_dir = save_dir

        create_nested_cv_datasets(
            metadata=metadata,
            num_outer_cv=self.num_outer_cv,
            num_inner_cv=self.num_inner_cv,
            target_column_name=target_column_name,
            test_size=test_size,
            val_size=val_size,
            stratify=stratify,
            random_state=random_state,
            save_dir=save_dir,
        )

    def get_dataset_path(self, outer_fold: int, inner_fold: int = None) -> Path:
        path = (
            self.save_dir / f"outer_fold={outer_fold}-inner_fold={inner_fold}.parquet"
            if inner_fold is not None
            else self.save_dir / f"outer_fold={outer_fold}.parquet"
        )
        return path

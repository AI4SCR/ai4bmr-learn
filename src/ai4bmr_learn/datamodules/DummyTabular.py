# %%
from pathlib import Path

from .Tabular import TabularDataModule


class DummyTabular(TabularDataModule):

    def __init__(
        self,
        data_path: Path = Path("~/data/ai4bmr-learn/DummyTabular/data.parquet").expanduser().resolve(),
        metadata_path: Path = Path("~/data/ai4bmr-learn/DummyTabular/metadata.parquet").expanduser().resolve(),
        target_column_name: str = "label",
        splits_path: Path = None,
        **kwargs
    ):
        super().__init__(
            data_path=data_path,
            metadata_path=metadata_path,
            splits_path=splits_path,
            target_column_name=target_column_name,
            **kwargs
        )

    def _prepare_data(self) -> None:
        # NOTE: here we load one of our datasets and bring it into the right format for the training that we want to do.
        from ai4bmr_datasets.datasets.DummyTabular import DummyTabular

        ds = DummyTabular(num_samples=1000, num_classes=2, num_features=10)
        data = ds.load()

        if not self.data_path.exists():
            self.data_path.parent.mkdir(parents=True, exist_ok=True)
            data["data"].to_parquet(self.data_path, engine='fastparquet')

        if not self.metadata_path.exists():
            self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
            data["metadata"].to_parquet(self.metadata_path, engine='fastparquet')

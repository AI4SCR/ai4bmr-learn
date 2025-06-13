# %%
from pathlib import Path
import lightning as L

class DummyImages(L.LightningDataModule):

    def __init__(
        self,
        data_path: Path = Path("~/data/ai4bmr-learn/DummyImages/data.parquet").expanduser().resolve(),
        metadata_path: Path = Path("~/data/ai4bmr-learn/DummyImages/metadata.parquet").expanduser().resolve(),
        target_column_name: str = "label",
        splits_path: Path = None,
        **kwargs
    ):
        pass

    def prepare_data(self) -> None:
        # NOTE: here we load one of our datasets and bring it into the right format for the training that we want to do.
        from ai4bmr_datasets.datasets.DummyImages import DummyImages

        ds = DummyImages(num_samples=1000, num_classes=2, num_channels=3, height=224, width=224)
        ds.setup()

        if not self.data_path.exists():
            self.data_path.parent.mkdir(parents=True, exist_ok=True)
            data["data"].to_parquet(self.data_path, engine='fastparquet')

        if not self.metadata_path.exists():
            self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
            data["metadata"].to_parquet(self.metadata_path, engine='fastparquet')

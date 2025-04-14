# %%
from pathlib import Path

from .MIL import MILDataModule


class DummyMIL(MILDataModule):

    def __init__(
        self,
        data_dir: Path = Path("~/data/ai4bmr-learn/DummyMIL/data/").expanduser().resolve(),
        metadata_path: Path = Path("~/data/ai4bmr-learn/DummyMIL/metadata.parquet").expanduser().resolve(),
        target_column_name: str = "target",
        splits_path: Path = None,
        val_size: float = 0.1,
        **kwargs,
    ):
        super().__init__(
            data_dir=data_dir,
            metadata_path=metadata_path,
            splits_path=splits_path,
            target_column_name=target_column_name,
            val_size=val_size,
            **kwargs,
        )

    def _prepare_data(self) -> None:
        # NOTE: here we load one of our datasets and bring it into the right format for the training that we want to do.
        import numpy as np
        import pandas as pd

        num_samples = 100
        num_features = 25
        num_classes = 2

        targets = pd.Categorical(np.random.choice(range(num_classes), num_samples))
        metadata = pd.DataFrame({self.target_column_name: targets})
        metadata.index.name = "sample_id"
        metadata.index = metadata.index.astype(str)
        metadata = metadata.convert_dtypes()

        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata.to_parquet(self.metadata_path, engine="fastparquet")

        self.data_dir.mkdir(parents=True, exist_ok=True)
        for i in range(num_samples):
            num_instances = np.random.randint(3, 10)
            data = np.random.randn(num_instances, num_features)
            data = data.astype(np.float32)
            data = pd.DataFrame(data)
            data.columns = data.columns.astype(str)
            data.to_parquet(self.data_dir / f"{i}.parquet", engine="fastparquet")

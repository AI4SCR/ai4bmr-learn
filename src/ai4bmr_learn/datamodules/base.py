# %%
from pathlib import Path
import lightning as L
import pandas as pd

# %% TABULAR
from ai4bmr_learn.datasets.Tabular import TabularDataset
class BaseTabularDataModule(L.LightningDataModule):

    def __init__(
            self,
            data_path: Path,
            metadata_path: Path,
            target_column_name: str = "target",
    ):
        super().__init__()

        # CONFIGURE PATHS
        self.data_path = data_path.resolve()
        self.metadata_path = metadata_path.resolve()
        self.target_column_name = target_column_name

        self.tabular = None

    def setup(self, stage=None):
        self.tabular = TabularDataset.from_paths(
            data_path=self.data_path,
            metadata_path=self.metadata_path,)

    def prepare_data(self):
        pass


# %%
from ai4bmr_learn.datasets.mil import MILDataset


class BaseMILDataModule(L.LightningDataModule):

    def __init__(
            self,
            data_dir: Path,
            metadata_path: Path,
            target_column_name: str = "target",
    ):
        super().__init__()

        # CONFIGURE PATHS
        self.data_dir = data_dir.resolve()
        self.metadata_path = metadata_path.resolve()
        self.target_column_name = target_column_name

        self.mil = None

    def setup(self, stage=None):
        data = {k: pd.read_parquet(self.data_dir / f"{k}.parquet", engine="fastparquet")
                for k in self.data_dir.glob('*.parquet')}

        metadata = pd.read_parquet(self.metadata_path, engine="fastparquet")
        self.mil = MILDataset(data=data, metadata=metadata, target_column_name=self.target_column_name)

    def prepare_data(self):
        pass


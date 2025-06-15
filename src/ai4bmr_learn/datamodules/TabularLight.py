# %%
from pathlib import Path

import lightning as L
import numpy as np
import pandas as pd
from torch import get_num_threads
from torch.utils.data import DataLoader

from ..data.splits import Split, generate_splits


class TabularDataModule(L.LightningDataModule):

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

    def setup(self, *args, **kwargs) -> None:
        from ai4bmr_learn.datasets.Tabular import TabularDataset

        self.tabular = TabularDataset.from_paths(data_path=str(self.data_path),
                                                 metadata_path=str(self.metadata_path),
                                                 target_column_name=self.target_column_name)

    def prepare_data(self) -> None:
        raise NotImplementedError()

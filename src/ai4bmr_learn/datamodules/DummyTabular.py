# %%
from pathlib import Path

from .Tabular import TabularDataModule


class DummyTabular:

    def __init__(self):
        self.tabular = None

    def prepare_data(self) -> None:
        pass

    def setup(self) -> None:
        from ai4bmr_datasets.datasets.DummyTabular import DummyTabular
        from ai4bmr_learn.datasets.Tabular import TabularDataset

        ds = DummyTabular(num_samples=1000, num_classes=2, num_features=10)

        self.tabular = TabularDataset(
            data=ds.data,
            metadata=ds.metadata,
            target_column_name='label'
        )



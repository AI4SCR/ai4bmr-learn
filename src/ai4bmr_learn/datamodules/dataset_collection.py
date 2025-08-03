import lightning as L
from torch.utils.data import DataLoader, Dataset
from torch import get_num_threads
from typing import Any

class DatasetCollection(L.LightningDataModule):

    def __init__(self,
                 datasets: dict[str, Dataset],
                 dataloaders: dict | None = None,
                 batch_size: int = 64,
                 num_workers: int = None,
                 persistent_workers: bool = True,
                 shuffle: bool = True,
                 pin_memory: bool = True,
                 collate_fn: Any | None = None
                 ):
        super().__init__()

        # DATASETS
        self.fit_set = datasets['fit'] if 'fit' in datasets else None
        self.val_set = datasets['val'] if 'val' in datasets else None
        self.test_set = datasets['test'] if 'test' in datasets else None

        # DATALOADERS
        self.dataloaders = dataloaders or {}

        num_workers = num_workers if num_workers is not None else max(0, get_num_threads() - 1)
        persistent_workers = persistent_workers if num_workers > 0 else False

        self.dl_kwargs = {
            'batch_size': batch_size,
            'num_workers': num_workers,
            'persistent_workers': persistent_workers,
            'shuffle': shuffle,
            'pin_memory': pin_memory,
            'collate_fn': collate_fn

        }

        self.save_hyperparameters()

    def setup(self, stage: str | None = None):

        # if stage == 'fit' or stage is None:
        if hasattr(self.fit_set, 'setup'):
            self.fit_set.setup()
        # if stage == 'validate' or stage is None:
        if hasattr(self.val_set, 'setup'):
            self.val_set.setup()
        # if stage == 'test' or stage is None:
        if hasattr(self.test_set, 'setup'):
            self.test_set.setup()

    def train_dataloader(self):
        kwargs = self.dl_kwargs.copy()
        kwargs.update(self.dataloaders.get('fit', {}))
        return DataLoader(self.fit_set, **kwargs)

    def val_dataloader(self):
        kwargs = self.dl_kwargs.copy()
        kwargs.update(self.dataloaders.get('val', {}))
        return DataLoader(self.val_set, **kwargs)

    def test_dataloader(self):
        kwargs = self.dl_kwargs.copy()
        kwargs.update(self.dataloaders.get('test', {}))
        return DataLoader(self.test_set, **kwargs)

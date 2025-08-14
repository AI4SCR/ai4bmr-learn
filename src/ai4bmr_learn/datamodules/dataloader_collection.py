import lightning as L
from torch.utils.data import DataLoader, Dataset
from torch import get_num_threads
from typing import Any

class DataLoaderCollection(L.LightningDataModule):

    def __init__(self, dataloaders: dict[str, list[DataLoader]]):
        super().__init__()
        self.dataloaders = dataloaders
        self.save_hyperparameters()

    def setup(self, stage: str | None = None):

        for stage, stage_loaders in self.dataloaders:
            for loader in stage_loaders:
                dataset = loader.dataset
                if hasattr(dataset, 'setup'):
                    dataset.setup()

    def train_dataloader(self):
        return self.dataloaders['train']

    def val_dataloader(self):
        return self.dataloaders['val']

    def test_dataloader(self):
        return self.dataloaders['test']

    def predict_dataloader(self):
        return self.dataloaders['predict']

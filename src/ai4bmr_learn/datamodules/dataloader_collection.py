import lightning as L
from torch.utils.data import Dataset
from typing import Callable
import torch.utils.data

class DataLoader(torch.utils.data.DataLoader):

    def __init__(self, dataset: Dataset, collate_fn: Callable = None, **kwargs):
        super().__init__(dataset, collate_fn=collate_fn, **kwargs)


class DataLoaderCollection(L.LightningDataModule):

    def __init__(self, dataloaders: dict[str, list[DataLoader]]):
        super().__init__()
        self.dataloaders = dataloaders
        for k,v in dataloaders.items():
            assert isinstance(v, list), f'Please provide the datasets as lists under each key, i.e fit: [dataset1]'
            assert len(v) == 1, f'Only one dataloader per split supported at this time.'
        self.save_hyperparameters()

    def setup(self, stage: str | None = None):

        for stage, stage_loaders in self.dataloaders.items():
            for loader in stage_loaders:
                dataset = loader.dataset
                if hasattr(dataset, 'setup'):
                    dataset.setup()

    def train_dataloader(self):
        return self.dataloaders['fit'][0]

    def val_dataloader(self):
        return self.dataloaders['val'][0]

    def test_dataloader(self):
        return self.dataloaders['test'][0]

    def predict_dataloader(self):
        return self.dataloaders['predict'][0]

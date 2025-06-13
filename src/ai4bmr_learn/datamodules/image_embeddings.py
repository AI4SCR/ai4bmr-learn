from pathlib import Path
from torch.utils.data import DataLoader, Dataset
import lightning as l
import torch
import pandas as pd
from loguru import logger
from torch import get_num_threads
from ai4bmr_learn.datamodules.TabularLight import TabularDataModule

class ImageEmbeddings(TabularDataModule):

    def __init__(self,
                 data_path: Path, metadata_path: Path,
                 dataset: Dataset,
                 backbone: l.LightningModule = None,
                 target_column_name: str = "target",
                 batch_size: int = 64,
                 num_workers: int = None,
                 persistent_workers: bool = True,
                 shuffle: bool = True,
                 pin_memory: bool = True
                 ):

        super().__init__(data_path=data_path, metadata_path=metadata_path, target_column_name=target_column_name)

        self.backbone = backbone
        self.dataset = dataset

        # DATALOADERS
        self.batch_size = batch_size
        self.num_workers = num_workers if num_workers is not None else max(0, get_num_threads() - 1)
        self.persistent_workers = persistent_workers if self.num_workers > 0 else False
        self.shuffle = shuffle
        self.pin_memory = pin_memory

    def setup(self):
        from ai4bmr_learn.datasets.Tabular import TabularDataset
        data = pd.read_parquet(self.data_path, engine="fastparquet")
        metadata = pd.read_parquet(self.metadata_path, engine="fastparquet")
        self.tabular = TabularDataset(data=data, metadata=metadata, target_column_name='target')

    def prepare_data(self):

        if self.data_path.exists() and self.metadata_path.exists():
            logger.info(f"Data already prepared.")
            return

        self.dataset.setup()

        dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            shuffle=self.shuffle,
            pin_memory=self.pin_memory
        )

        trainer = l.Trainer()
        preds = trainer.predict(self.backbone, dataloader)

        data = []
        metadata = []
        for batch in preds:
            data.append(pd.DataFrame(batch['embedding']))
            metadata.append(pd.DataFrame(batch['target'], columns=['target']))

        data = pd.concat(data)
        data = data.reset_index(drop=True)
        data = data.convert_dtypes()
        data.columns = [str(i) for i in data.columns]
        self.data_path.parent.mkdir(parents=True, exist_ok=True)
        data.to_parquet(self.data_path, engine="fastparquet")

        metadata = pd.concat(metadata)
        metadata = metadata.reset_index(drop=True)
        metadata = metadata.convert_dtypes()
        metadata = metadata.astype({'target': 'category'})
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata.to_parquet(self.metadata_path, engine='fastparquet')
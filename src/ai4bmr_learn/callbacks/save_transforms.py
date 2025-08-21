from typing import Any

from lightning.pytorch.callbacks import Callback
from pathlib import Path
import shutil
import torch
from loguru import logger
import pickle

class SaveTransforms(Callback):

    def __init__(self, save_dir: Path | None = None, overwrite: bool = False) -> None:

        if save_dir is not None and save_dir.exists() and not overwrite:
            raise ValueError(f'`save_dir already exists. Use `overwrite=True`. {save_dir}')
        self.save_dir = save_dir

    def get_save_dir(self, trainer):
        save_dir = self.save_dir or Path(trainer.logger.save_dir) / trainer.logger.experiment.project / trainer.logger.experiment._attach_id / 'transforms'
        return save_dir

    def save_loaders(self, loaders: list, save_dir: Path):
        save_dir.mkdir(parents=True, exist_ok=True)
        for name, loader in loaders:
            if loader is None:
                continue
            transform = loader.dataset.transform
            if transform is not None:
                save_path = save_dir / f'{name}.pkl'
                save_dir.mkdir(parents=True, exist_ok=True)
                with open(save_path, 'wb') as f:
                    pickle.dump(transform, f)

    def on_train_epoch_start(self, trainer, pl_module) -> None:

        save_dir = self.get_save_dir(trainer)
        assert trainer.train_dataloader is not None

        loaders = [('train', trainer.train_dataloader)]
        self.save_loaders(loaders, save_dir)

    def on_validation_epoch_start(self, trainer, pl_module) -> None:

        save_dir = self.get_save_dir(trainer)
        assert trainer.val_dataloaders is not None

        loaders = trainer.val_dataloaders
        if isinstance(loaders, list):
            loaders = [(f'val-dl_idx={i}', loader) for i, loader in enumerate(loaders)]
        else:
            loaders = [('val', loaders)]
        self.save_loaders(loaders, save_dir)

    def on_test_epoch_start(self, trainer, pl_module) -> None:

        save_dir = self.get_save_dir(trainer)
        assert trainer.test_dataloaders is not None

        loaders = trainer.test_dataloaders
        if isinstance(loaders, list):
            loaders = [(f'test-dl_idx={i}', loader) for i, loader in enumerate(loaders)]
        else:
            loaders = [('test', loaders)]
        self.save_loaders(loaders, save_dir)

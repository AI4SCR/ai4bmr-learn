from typing import Any

from lightning.pytorch.callbacks import Callback
from pathlib import Path
import shutil
import torch

class SavePredictions(Callback):

    def __init__(self, save_dir: Path, save_key: str | None = None, drop_keys: list[str] | None = None, force: bool = False) -> None:

        if save_dir.exists() and not force:
            raise ValueError(f'Directory "{save_dir}" already exists. Use `force==True` to overwrite.')
        else:
            shutil.rmtree(save_dir, ignore_errors=True)

        self.save_dir = save_dir
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.save_key = save_key
        self.drop_keys = drop_keys

    def on_predict_batch_end(self, trainer, pl_module, outputs: Any, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        if self.save_key:
            data = batch
            data[self.save_key] = outputs
        else:
            data = outputs

        if self.drop_keys:
            for key in self.drop_keys:
                del data[key]

        save_path = self.save_dir / f'dl_idx={dataloader_idx}-batch_idx={batch_idx}.pt'
        torch.save(data, save_path)
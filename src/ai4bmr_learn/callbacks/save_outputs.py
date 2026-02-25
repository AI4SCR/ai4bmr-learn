from typing import Any

from lightning.pytorch.callbacks import Callback
from pathlib import Path
import shutil
import torch
from loguru import logger

def detach_and_to_cpu(item: Any) -> Any:
    if isinstance(item, torch.Tensor):
        return item.detach().cpu()
    elif isinstance(item, dict):
        return {k: detach_and_to_cpu(v) for k, v in item.items()}
    elif isinstance(item, list):
        return [detach_and_to_cpu(v) for v in item]
    elif isinstance(item, tuple):
        return tuple(detach_and_to_cpu(v) for v in item)
    else:
        return item

class SaveOutputs(Callback):

    def __init__(self, step: str, save_dir: Path, save_key: str | None = None, drop_keys: list[str] | None = None, force: bool = False) -> None:

        self.step = step
        assert step in ['fit', 'val', 'test', 'predict'], f'`step` must be one of {["fit", "val", "test", "predict"]}'

        if save_dir.exists() and not force:
            raise ValueError(f'Directory "{save_dir}" already exists. Use `force==True` to overwrite.')
        else:
            shutil.rmtree(save_dir, ignore_errors=True)

        self.save_dir = save_dir

        self.save_key = save_key
        self.drop_keys = drop_keys

    def run(self, batch, outputs, batch_idx, dataloader_idx):
        self.save_dir.mkdir(parents=True, exist_ok=True)

        if self.save_key:
            data = batch
            data[self.save_key] = outputs
        else:
            data = outputs

        if self.drop_keys:
            for key in self.drop_keys:
                assert key in data, f'Key "{key}" not found in batch. Available keys: {list(data.keys())}'
                del data[key]

        data = detach_and_to_cpu(data)
        save_path = self.save_dir / f'dl_idx={dataloader_idx}-batch_idx={batch_idx}.pt'
        torch.save(data, save_path)

    def on_train_batch_end(self, trainer, pl_module, outputs: Any, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        if self.step == 'fit' and not trainer.sanity_checking:
            self.run(batch=batch, outputs=outputs, batch_idx=batch_idx, dataloader_idx=dataloader_idx)

    def on_validation_batch_end(self, trainer, pl_module, outputs: Any, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        if self.step == 'val' and not trainer.sanity_checking:
            self.run(batch=batch, outputs=outputs, batch_idx=batch_idx, dataloader_idx=dataloader_idx)

    def on_test_batch_end(self, trainer, pl_module, outputs: Any, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        if self.step == 'test' and not trainer.sanity_checking:
            self.run(batch=batch, outputs=outputs, batch_idx=batch_idx, dataloader_idx=dataloader_idx)

    def on_predict_batch_end(self, trainer, pl_module, outputs: Any, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        if self.step == 'predict' and not trainer.sanity_checking:
            self.run(batch=batch, outputs=outputs, batch_idx=batch_idx, dataloader_idx=dataloader_idx)

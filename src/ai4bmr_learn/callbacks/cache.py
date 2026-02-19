from holoviews import output
from lightning.pytorch.callbacks import Callback
from loguru import logger
import pickle
from pathlib import Path
import glom
from sympy.printing.pytorch import torch


def move_to_cpu(output):
    for key in output:
        if isinstance(output[key], torch.Tensor):
            output[key] = output[key].detach().cpu()
        if isinstance(output[key], dict):
            move_to_cpu(output[key])

class Cache(Callback):
    name: str = 'cache'

    def __init__(self, num_batches: int | None = None, save_dir: Path | None = None, fname: str | None = None,
                 exclude_keys: list[str] | None = None, ignore_missing: bool = False, save: bool = True):
        super().__init__()

        fname = fname or self.name
        self.fname = f'{fname}.pkl'
        self.num_batches = num_batches
        self.outputs = []
        self.exclude_keys = exclude_keys or []
        self.ignore_missing = ignore_missing
        self.save = save

        assert save or save_dir is None, "Cannot specify save_dir if save is False."

        if save_dir is not None:
            self.save_dir = Path(save_dir).expanduser().resolve()
        self.save_dir = save_dir

        logger.info(f'`{self.name.capitalize()}Cache` attached')


    def configure_save_dir(self, trainer):
        if trainer.fast_dev_run or not self.save:
            return

        if self.save_dir is not None:
            pass
        elif trainer.logger.save_dir is not None:
            save_dir = trainer.logger.save_dir
            name = trainer.logger.name
            experiment_id = trainer.logger.experiment.id

            self.save_dir = Path(save_dir) / name / experiment_id / "cache"
        else:
            logger.warning(f"Cache has no save_dir configured, outputs will not be saved to disk.")
        logger.info(f"Cache will be saved to {self.save_dir / self.fname}")

    def delete_keys(self, output):
        for key in self.exclude_keys:
            _ = glom.delete(output, key, ignore_missing=self.ignore_missing)

    def accumulate(self, outputs):
        accumulate = (self.num_batches is None) or (len(self.outputs) < self.num_batches)

        if accumulate:
            self.delete_keys(outputs)
            move_to_cpu(outputs)
            self.outputs.append(outputs)

    def reset(self):
        self.outputs = []

    def save_to_disk(self):
        if not self.save or self.save_dir is None:
            return

        save_path = self.save_dir / self.fname
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(self.outputs, f)


class TrainCache(Cache):
    name: str = 'train'

    def on_train_start(self, trainer, pl_module) -> None:
        self.configure_save_dir(trainer=trainer)

    def on_train_epoch_start(self, trainer, pl_module) -> None:
        self.reset()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self.accumulate(outputs)

    def on_train_end(self, trainer, pl_module) -> None:
        self.save_to_disk()


class ValidationCache(Cache):
    name: str = 'validation'

    def on_validation_start(self, trainer, pl_module) -> None:
        self.configure_save_dir(trainer=trainer)

    def on_validation_epoch_start(self, trainer, pl_module) -> None:
        self.reset()

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self.accumulate(outputs)

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        self.save_to_disk()


class TestCache(Cache):
    name: str = 'test'

    def on_test_start(self, trainer, pl_module) -> None:
        self.configure_save_dir(trainer=trainer)

    def on_test_epoch_start(self, trainer, pl_module) -> None:
        self.reset()

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self.accumulate(outputs)

    def on_test_end(self, trainer, pl_module) -> None:
        self.save_to_disk()

class PredictionCache(Cache):
    name: str = 'prediction'

    def on_predict_start(self, trainer, pl_module) -> None:
        self.configure_save_dir(trainer=trainer)

    def on_predict_epoch_start(self, trainer, pl_module) -> None:
        self.reset()

    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self.accumulate(outputs)

    def on_predict_end(self, trainer, pl_module) -> None:
        self.save_to_disk()

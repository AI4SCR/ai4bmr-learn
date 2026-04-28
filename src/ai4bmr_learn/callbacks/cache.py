from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import glom
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import Callback
from loguru import logger


def move_to_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: move_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [move_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(move_to_cpu(item) for item in value)
    return value


class Cache(Callback):
    name = "cache"

    def __init__(
        self,
        num_batches: int | None = None,
        save_dir: Path | None = None,
        name: str | None = None,
        include_keys: list[str] | None = None,
        exclude_keys: list[str] | None = None,
        ignore_missing: bool = False,
        save: bool = True,
        save_in_batches: bool = False,
    ) -> None:
        super().__init__()
        assert save or save_dir is None, "save_dir requires save=True"
        assert not save_in_batches or save, "save_in_batches requires save=True"
        assert not (include_keys and exclude_keys), "include_keys and exclude_keys are mutually exclusive"

        self.num_batches = num_batches
        self.outputs: list[dict[str, Any]] = []
        self.include_keys = include_keys or []
        self.exclude_keys = exclude_keys or []
        self.ignore_missing = ignore_missing
        self.save = save
        self.save_in_batches = save_in_batches
        self._batch_counter: int = 0
        self.name = name or self.name
        self.save_dir = Path(save_dir).expanduser().resolve() if save_dir else None

        logger.info(f"`{self.name.capitalize()}Cache` attached")

    def configure_save_dir(self, trainer: Trainer) -> None:
        if trainer.fast_dev_run or not self.save:
            return

        if self.save_dir is None:
            logger_ = trainer.logger
            assert logger_ is not None, "Cache requires a trainer logger or an explicit save_dir"
            assert logger_.save_dir is not None, "Cache logger must define save_dir"
            assert hasattr(logger_, "name"), "Cache logger must define name"
            assert hasattr(logger_, "experiment"), "Cache logger must expose experiment"
            experiment_id = logger_.experiment.id
            self.save_dir = Path(logger_.save_dir) / logger_.name / experiment_id / "cache"

        if self.save_in_batches:
            logger.info(f"Cache will be saved to {self.save_dir / self.name}/<batch_idx>.pt")
        else:
            logger.info(f"Cache will be saved to {self.save_dir / self.name}.pt")

    def reset(self) -> None:
        self.outputs = []
        self._batch_counter = 0

    def include_selected_keys(self, output: dict[str, Any]) -> dict[str, Any]:
        included: dict[str, Any] = {}
        for key in self.include_keys:
            try:
                value = glom.glom(output, key)
            except glom.PathAccessError:
                if self.ignore_missing:
                    continue
                raise
            glom.assign(included, key, value, missing=dict)
        return included

    def delete_excluded_keys(self, output: dict[str, Any]) -> None:
        for key in self.exclude_keys:
            glom.delete(output, key, ignore_missing=self.ignore_missing)

    def accumulate(self, output: dict[str, Any]) -> None:
        if self.num_batches is not None and len(self.outputs) >= self.num_batches:
            return

        cached_output = deepcopy(output)
        if self.include_keys:
            cached_output = self.include_selected_keys(cached_output)
        else:
            self.delete_excluded_keys(cached_output)
        self.outputs.append(move_to_cpu(cached_output))
        if self.save_in_batches:
            self._save_batch()

    def _save_batch(self) -> None:
        if not self.save or self.save_dir is None:
            return
        save_path = self.save_dir / self.name / f"{self._batch_counter:06d}.pt"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.outputs, save_path)
        self._batch_counter += 1
        self.outputs = []

    def save_to_disk(self) -> None:
        if not self.save or self.save_dir is None or not self.outputs:
            return

        save_path = self.save_dir / f"{self.name}.pt"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.outputs, save_path)


class TestCache(Cache):
    name = "test"

    def on_test_start(self, trainer: Trainer, pl_module) -> None:
        self.configure_save_dir(trainer)

    def on_test_epoch_start(self, trainer: Trainer, pl_module) -> None:
        self.reset()

    def on_test_batch_end(self, trainer: Trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0) -> None:
        self.accumulate(outputs)

    def on_test_end(self, trainer: Trainer, pl_module) -> None:
        self.save_to_disk()


class TrainCache(Cache):
    name = "train"

    def on_train_start(self, trainer: Trainer, pl_module) -> None:
        self.configure_save_dir(trainer)

    def on_train_epoch_start(self, trainer: Trainer, pl_module) -> None:
        self.reset()

    def on_train_batch_end(self, trainer: Trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0) -> None:
        self.accumulate(outputs)

    def on_train_end(self, trainer: Trainer, pl_module) -> None:
        self.save_to_disk()


class ValidationCache(Cache):
    name = "validation"

    def on_validation_start(self, trainer: Trainer, pl_module) -> None:
        self.configure_save_dir(trainer)

    def on_validation_epoch_start(self, trainer: Trainer, pl_module) -> None:
        self.reset()

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module,
        outputs,
        batch,
        batch_idx,
        dataloader_idx=0,
    ) -> None:
        self.accumulate(outputs)

    def on_validation_epoch_end(self, trainer: Trainer, pl_module) -> None:
        self.save_to_disk()


class PredictionCache(Cache):
    name = "prediction"

    def on_predict_start(self, trainer: Trainer, pl_module) -> None:
        self.configure_save_dir(trainer)

    def on_predict_epoch_start(self, trainer: Trainer, pl_module) -> None:
        self.reset()

    def on_predict_batch_end(self, trainer: Trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0) -> None:
        self.accumulate(outputs)

    def on_predict_end(self, trainer: Trainer, pl_module) -> None:
        self.save_to_disk()

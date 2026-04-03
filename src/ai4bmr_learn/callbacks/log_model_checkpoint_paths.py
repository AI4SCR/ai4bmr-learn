from __future__ import annotations

import lightning as L
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from loguru import logger


class LogCheckpointPathsCallback(Callback):
    def on_train_end(self, trainer: L.Trainer, pl_module) -> None:
        self.log_checkpoint_paths(trainer)

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module) -> None:
        self.log_checkpoint_paths(trainer)

    def log_checkpoint_paths(self, trainer: L.Trainer) -> None:
        if trainer.fast_dev_run:
            return

        logger_ = trainer.logger
        if logger_ is None or not hasattr(logger_, "experiment"):
            return

        checkpoint_callbacks = [callback for callback in trainer.callbacks if isinstance(callback, ModelCheckpoint)]
        if not checkpoint_callbacks:
            logger.warning("LogCheckpointPathsCallback requires a ModelCheckpoint callback")
            return

        for callback in checkpoint_callbacks:
            update: dict[str, str] = {}
            if callback.best_model_path and callback.monitor is not None:
                update["best_model_path"] = callback.best_model_path
            if callback.last_model_path and callback.monitor is None:
                update["last_model_path"] = callback.last_model_path
            if update:
                logger_.experiment.config.update(update, allow_val_change=True)

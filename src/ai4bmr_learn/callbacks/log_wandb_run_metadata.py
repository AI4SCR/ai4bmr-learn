from __future__ import annotations

import lightning as L
from lightning.pytorch.callbacks import Callback


class LogWandbRunMetadataCallback(Callback):
    def __init__(self) -> None:
        super().__init__()
        self.completed = False

    def on_fit_start(self, trainer: L.Trainer, pl_module) -> None:
        self.log_run_metadata(trainer)

    def on_test_start(self, trainer: L.Trainer, pl_module) -> None:
        self.log_run_metadata(trainer)

    def on_predict_start(self, trainer: L.Trainer, pl_module) -> None:
        self.log_run_metadata(trainer)

    def log_run_metadata(self, trainer: L.Trainer) -> None:
        if trainer.fast_dev_run or self.completed:
            return

        logger = trainer.logger
        if logger is None or not hasattr(logger, "experiment"):
            return

        experiment = logger.experiment
        experiment.config.update(
            {
                "run_id": experiment.id,
                "run_name": experiment.name,
            },
            allow_val_change=False,
        )
        self.completed = True

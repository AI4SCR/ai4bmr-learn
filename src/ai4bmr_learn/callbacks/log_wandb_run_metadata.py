import lightning as L
from lightning.pytorch.callbacks import Callback

class LogWandbRunMetadataCallback(Callback):
    def on_fit_start(self, trainer: L.Trainer, pl_module):

        if trainer.fast_dev_run:
            return

        logger = trainer.logger
        if logger is None or not hasattr(logger, "experiment"):
            return

        experiment = logger.experiment

        run_id = experiment.id
        run_name = experiment.name

        experiment.config.update(
            {
                "run_id": run_id,
                "run_name": run_name,
            },
            allow_val_change=False,
        )

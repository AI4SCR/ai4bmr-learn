import lightning as L
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from loguru import logger

class LogCheckpointPathsCallback(Callback):
    def on_fit_end(self, trainer: L.Trainer, pl_module):

        if trainer.fast_dev_run:
            return

        ckpt_cbs = [
            cb for cb in trainer.callbacks
            if isinstance(cb, ModelCheckpoint)
        ]
        if not ckpt_cbs:
            logger.warning("You provided `LogCheckpointPathsCallback` but no ModelCheckpoint callback found in trainer.callbacks")
            return

        ckpt_cb = ckpt_cbs[0]

        best_model_path = ckpt_cb.best_model_path
        last_model_path = ckpt_cb.last_model_path

        logger = trainer.logger
        if logger is None or not hasattr(logger, "experiment"):
            return

        logger.experiment.config.update(
            {
                "best_model_path": best_model_path,
                "last_model_path": last_model_path,
            },
            allow_val_change=True,
        )
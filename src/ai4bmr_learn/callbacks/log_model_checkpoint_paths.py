import lightning as L
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from loguru import logger

class LogCheckpointPathsCallback(Callback):

    def on_train_end(self, trainer, pl_module):
        self.log_checkpoint_paths(trainer=trainer, pl_module=pl_module)

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module):
        self.log_checkpoint_paths(trainer=trainer, pl_module=pl_module)

    def log_checkpoint_paths(self, trainer: L.Trainer, pl_module):

        if trainer.fast_dev_run:
            return

        ckpt_cbs = [
            cb for cb in trainer.callbacks
            if isinstance(cb, ModelCheckpoint)
        ]
        if not ckpt_cbs:
            logger.warning("You provided `LogCheckpointPathsCallback` but no ModelCheckpoint callback found in trainer.callbacks")
            return


        for cb in ckpt_cbs:

            upt = {}
            if cb.best_model_path and cb.monitor is not None:
                upt["best_model_path"] = cb.best_model_path
            if cb.last_model_path and cb.monitor is None:
                upt["last_model_path"] = cb.last_model_path

            logger = trainer.logger
            if logger is None or not hasattr(logger, "experiment"):
                return

            logger.experiment.config.update(upt, allow_val_change=True)
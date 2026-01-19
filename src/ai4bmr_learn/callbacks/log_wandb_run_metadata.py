import lightning as L
from lightning.pytorch.callbacks import Callback

class LogWandbRunMetadataCallback(Callback):

    def __init__(self):
        super().__init__()
        self.completed = False

    def on_fit_start(self, trainer: L.Trainer, pl_module):
        self.log_run_metadata(trainer=trainer, pl_module=pl_module)

    def on_test_start(self, trainer: L.Trainer, pl_module):
        self.log_run_metadata(trainer=trainer, pl_module=pl_module)

    def on_predict_start(self, trainer: L.Trainer, pl_module):
        self.log_run_metadata(trainer=trainer, pl_module=pl_module)

    def log_run_metadata(self, trainer: L.Trainer, pl_module):

        if trainer.fast_dev_run or self.completed:
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

        self.completed = True

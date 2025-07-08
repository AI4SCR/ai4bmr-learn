from lightning.pytorch.callbacks import Callback
from copy import deepcopy

default_keys = [
        "accumulate_grad_batches",
        "check_val_every_n_epoch",
        "reload_dataloaders_every_n_epochs",
        "gradient_clip_val",
        "gradient_clip_algorithm",
        "log_every_n_steps",
        "fast_dev_run",
        "overfit_batches",
        "limit_train_batches",
        "limit_val_batches",
        "limit_test_batches",
        "limit_predict_batches",
        "num_sanity_val_steps",
        "val_check_interval",
]


class LogTrainer(Callback):

    def __init__(self, keys: list[str] = default_keys, prefix: str = "trainer."):
        self.keys = keys
        self.prefix = prefix

    def on_fit_start(self, trainer, pl_module):
        trainer_state = {f'{self.prefix}{k}':trainer.__dict__[k] for k in self.keys}
        trainer.logger.experiment.config.update(trainer_state)

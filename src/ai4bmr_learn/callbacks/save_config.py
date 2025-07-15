from lightning.pytorch.cli import SaveConfigCallback
from lightning.pytorch.loggers import Logger
import lightning as L
from jsonargparse._namespace import Namespace

def to_dict(item):
    item = vars(item) if isinstance(item, Namespace) else item

    if isinstance(item, dict):
        return {k: to_dict(v) for k, v in item.items()}

    if isinstance(item, (list, tuple)):
        return [to_dict(i) for i in item]

    return item

class LoggerSaveConfigCallback(SaveConfigCallback):
    def save_config(self, trainer: L.Trainer, pl_module: L.LightningModule, stage: str) -> None:
        if isinstance(trainer.logger, Logger):
            config = to_dict(self.config)
            trainer.logger.experiment.config.update(config)

from lightning.pytorch.cli import SaveConfigCallback
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import Logger
import lightning as L
from pathlib import Path
from loguru import logger

from ai4bmr_learn.utils.utils import to_dict


class LoggerSaveConfigCallback(SaveConfigCallback):
    def save_config(self, trainer: L.Trainer, pl_module: L.LightningModule, stage: str) -> None:
        if isinstance(trainer.logger, Logger):
            config = to_dict(self.config)
            experiment = getattr(trainer.logger, "experiment", None)
            save_dir = getattr(trainer.logger, "save_dir", None)
            project = getattr(experiment, "project", None)
            attach_id = getattr(experiment, "_attach_id", None)
            if experiment is None or save_dir is None or project is None or attach_id is None:
                logger.warning("Config could not be saved with the active logger.")
                return

            run_dir = Path(save_dir) / project / attach_id
            config["run_dir"] = str(run_dir)
            experiment.config.update(config)


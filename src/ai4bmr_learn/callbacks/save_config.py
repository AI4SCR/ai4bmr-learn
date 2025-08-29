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

            try:
                run_dir = Path(trainer.logger.save_dir) / trainer.logger.experiment.project / trainer.logger.experiment._attach_id
                config['run_dir'] = str(run_dir)

                trainer.logger.experiment.config.update(config)
            except AttributeError:
                logger.warning(f'Config could not be saved with used logger')



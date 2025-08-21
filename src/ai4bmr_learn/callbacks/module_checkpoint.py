from lightning.pytorch.callbacks import Callback
import torch
from pathlib import Path
from loguru import logger

class ModuleCheckpoint(Callback):
    def __init__(self, module_name: str, every_n_epochs: int = 1):
        self.module_name = module_name
        self.every_n_epochs = every_n_epochs

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.fast_dev_run:
            return

        epoch = trainer.current_epoch
        if (epoch + 1) % self.every_n_epochs != 0:
            return

        save_dir = Path(trainer.logger.save_dir) / trainer.logger.experiment.project / trainer.logger.experiment._attach_id / 'checkpoints'
        save_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving checkpoint for module '{self.module_name}' at epoch {epoch + 1} to {save_dir}")

        path = save_dir / f"module={self.module_name}-epoch={epoch}-step={trainer.global_step}.ckpt"

        module = pl_module
        for attr in self.module_name.split("."):
            module = getattr(module, attr)

        # TODO: move to CPU before saving
        torch.save(module.state_dict(), path)

        # key = f'{self.module_name}_ckpt_path'
        # trainer.logger.experiment.config.update({key: str(path)})

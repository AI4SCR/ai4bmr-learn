from lightning.pytorch.callbacks import Callback
from loguru import logger

class RunValidationOnTrainStart(Callback):

    def __init__(self):
        pass

    def on_fit_start(self, trainer, pl_module):
        # state = trainer.state
        # val_dataloaders=trainer.datamodule.val_dataloader()
        # trainer.validate(model=pl_module, dataloaders=val_dataloaders)
        # trainer.state = state
        stage = trainer.state.stage
        trainer.validating = True
        trainer._run_stage()
        trainer.state.stage = stage
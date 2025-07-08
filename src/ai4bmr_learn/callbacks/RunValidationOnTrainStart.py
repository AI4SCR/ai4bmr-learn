from lightning.pytorch.callbacks import Callback
from loguru import logger

class RunValidationOnTrainStart(Callback):

    def __init__(self):
        pass

    def on_train_start(self, trainer, pl_module):
        stage = trainer.state.stage
        trainer.validating = True
        trainer._run_stage()
        trainer.state.stage = stage

from lightning.pytorch.callbacks import Callback
from loguru import logger

class Debug(Callback):

    def __init__(self):
        pass

    def on_train_start(self, trainer, pl_module):
        logger.info('Running on_train_start')

    def on_train_end(self, trainer, pl_module):
        logger.info('Running on_train_start')

    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
        logger.info(f'Running on_validation_batch_start, [epoch={trainer.current_epoch}, is_last_batch={trainer.is_last_batch}, batch_idx={batch_idx}]')

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        logger.info(f'Running on_validation_batch_end, [epoch={trainer.current_epoch}, is_last_batch={trainer.is_last_batch}, batch_idx={batch_idx}, outputs_keys: {outputs.keys()}]')


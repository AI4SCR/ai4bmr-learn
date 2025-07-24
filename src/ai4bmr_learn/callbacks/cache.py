from lightning.pytorch.callbacks import Callback

class Cache(Callback):

    def __init__(self, num_samples: int | None):
        self.num_samples = num_samples
        self.outputs = []

    def accumulate(self, outputs):
        accumulate = (self.num_samples is None) or (len(self.outputs) < self.num_samples)

        if accumulate:
            self.outputs.append(outputs)

    def reset(self):
        self.outputs = []

class TrainCache(Cache):

    def on_train_start(self, trainer, pl_module) -> None:
        self.reset()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self.accumulate(outputs)


class ValidationCache(Cache):

    def on_validation_epoch_start(self, trainer, pl_module) -> None:
        self.reset()

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self.accumulate(outputs)


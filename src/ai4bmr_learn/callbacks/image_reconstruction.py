import numpy as np
import torch
from glom import glom
from lightning.pytorch.callbacks import Callback
from loguru import logger
from torchvision.utils import make_grid
from tqdm import tqdm

from ai4bmr_learn.utils.device import batch_to_device


class ImageReconstruction(Callback):

    def __init__(self,
                 run_before_train: bool = True, run_after_train: bool = True, run_every_num_epochs: int = 1,
                 image_key: str = 'image', prediction_key: str = 'mae.prediction', masks_key: str = 'mae.masks',
                 num_samples: int = 5, seed: int = 0, padding: int = 2):

        self.run_before_train = run_before_train
        self.run_after_train = run_after_train
        self.run_every_num_epochs = run_every_num_epochs

        self.image_key = image_key
        self.prediction_key = prediction_key
        self.masks_key = masks_key

        self.num_samples = num_samples
        self.padding = padding
        self.rng = np.random.default_rng(seed=seed)

        self.images = self.predictions = self.masks = None

    def accumulate(self, outputs) -> bool:
        accumulate = (self.num_samples is None) or self.images is None or (len(self.images) < self.num_samples)

        # a = outputs['mae']['prediction']
        # b = outputs['mae']['prediction_masked']
        # c = outputs['image']
        # (a == b).any()
        # (a == c).any()
        # (b == c).any()

        if accumulate:
            if self.images is None:
                self.images = glom(outputs, self.image_key)
            else:
                self.images = torch.vstack((self.images, glom(outputs, self.image_key)))

            if self.predictions is None:
                self.predictions = glom(outputs, self.prediction_key)
            else:
                self.predictions = torch.vstack((self.predictions, glom(outputs, self.prediction_key)))

            if self.masks is None:
                self.masks = glom(outputs, self.masks_key)
            else:
                self.masks = torch.vstack((self.masks, glom(outputs, self.masks_key)))
        else:
            return True
        return False

    def visualize(self, trainer):
        grid = []
        for img, pred in zip(self.images[:self.num_samples], self.predictions[:self.num_samples]):
            grid.extend([img, pred])
        grid = make_grid(grid, nrow=2, normalize=True, padding=self.padding)
        trainer.logger.log_image(key=f"mae/predictions", images=[grid])

    def on_train_start(self, trainer, pl_module):
        if self.run_before_train:
            logger.info(f'Visualize reconstructions [on_train_start]')
            outputs, batch_idx = self.get_outputs(trainer, pl_module)
            self.accumulate(outputs)
            self.visualize(trainer=trainer)
            self.reset()

    def on_train_end(self, trainer, pl_module):
        if self.run_after_train:
            logger.info(f'Visualize reconstructions [on_train_end]')
            outputs, batch_idx = self.get_outputs(trainer, pl_module)
            self.accumulate(outputs)
            self.visualize(trainer=trainer)
            self.reset()

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if not self.should_run(trainer=trainer):
            return

        is_accumulated = self.accumulate(outputs)
        # note: trainer.is_last_batch is not always reset for some reason
        is_last_batch = len(trainer.val_dataloaders) - 1 == batch_idx

        if is_last_batch:
            logger.info(f'Logging MAE predictions [num_samples={self.num_samples}]')
            self.visualize(trainer)
            self.reset()

    def get_outputs(self, trainer, pl_module):
        dl = trainer.val_dataloaders
        assert pl_module.device.type == 'cuda'
        with torch.no_grad():
            for batch_idx, batch in tqdm(enumerate(dl), desc='collecting outputs'):
                batch = batch_to_device(batch, pl_module.device)
                outputs = pl_module.validation_step(batch=batch, batch_idx=batch_idx)
                is_accumulated = self.accumulate(outputs)

                if is_accumulated or trainer.fast_dev_run:
                    break

        return outputs, batch_idx

    def should_run(self, trainer, force: bool = False):
        if trainer.sanity_checking:
            return False

        if force:
            return True

        # if trainer.current_epoch == 0 and self.run_before_train:
        #     return False

        if trainer.current_epoch % self.run_every_num_epochs == 0:
            return True

        return False

    def reset(self):
        self.images = self.predictions = self.masks = None
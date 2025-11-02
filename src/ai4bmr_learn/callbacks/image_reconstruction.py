import numpy as np
import torch
from glom import glom
from lightning.pytorch.callbacks import Callback
from loguru import logger
from torchvision.utils import make_grid

from ai4bmr_learn.callbacks.cache import ValidationCache


class ImageReconstructionSamples(Callback):

    def __init__(self,
                 image_key: str = 'image', prediction_key: str = 'mae.prediction', masks_key: str = 'mae.masks',
                 num_samples: int = 5, seed: int = 0, padding: int = 2):

        self.image_key = image_key
        self.prediction_key = prediction_key
        self.masks_key = masks_key

        self.num_samples = num_samples
        self.padding = padding
        self.rng = np.random.default_rng(seed=seed)

    def visualize(self, images, predictions, masks, trainer):

        grid = []
        for img, pred, mask in zip(images, predictions, masks):
            grid.extend([img, pred])
        grid = make_grid(grid, nrow=2, normalize=True, padding=self.padding)
        if not trainer.fast_dev_run:
            trainer.logger.log_image(key=f"mae/predictions", images=[grid])

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return

        cache, = list(filter(lambda x: isinstance(x, ValidationCache), trainer.callbacks))
        images, predictions, masks = self.get_data(cache=cache)
        logger.info(f'Image Reconstructions [num_samples={images.shape[0]}, epoch={trainer.current_epoch}]')
        self.visualize(images=images, predictions=predictions, masks=masks, trainer=trainer)

    def get_data(self, cache: ValidationCache):

        batch_size = glom(cache.outputs[0], self.image_key).shape[0]
        num_iter = np.ceil(self.num_samples / batch_size)

        images = []
        predictions = []
        masks = []
        for i, batch in enumerate(cache.outputs):
            images.append(glom(batch, self.image_key))
            predictions.append(glom(batch, self.prediction_key))
            masks.append(glom(batch, self.masks_key))

            if i == num_iter:
                break

        images = torch.cat(images, dim=0)[:self.num_samples]
        predictions = torch.cat(predictions, dim=0)[:self.num_samples]
        masks = torch.cat(masks, dim=0)[:self.num_samples]

        return images, predictions, masks

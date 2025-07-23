from lightning.pytorch.callbacks import Callback
import numpy as np
from torchvision.utils import make_grid
from loguru import logger

class ImageSamples(Callback):

    def __init__(self, num_samples: int = 5, seed: int = 0, padding: int = 2):

        self.num_samples = num_samples
        self.padding = padding
        self.rng = np.random.default_rng(seed=seed)

    def visualize(self, trainer, name: str, dataset):
        # random indices
        idc = np.arange(len(dataset))
        self.rng.shuffle(idc)
        idc = idc[:self.num_samples]

        items = [dataset[i] for i in idc]
        images = [i['image'] for i in items]

        grid = make_grid(images, normalize=True, padding=self.padding)

        trainer.logger.log_image(key=f"image_samples/{name}", images=[grid])

    def on_validation_start(self, trainer, pl_module) -> None:
        if trainer.current_epoch == 0 and not trainer.fast_dev_run:
            logger.info(f'Logging image samples from val [num_samples={self.num_samples}]')
            ds = trainer.val_dataloaders.dataset
            self.visualize(trainer=trainer, name='val', dataset=ds)

    def on_train_start(self, trainer, pl_module):
        if trainer.fast_dev_run:
            return

        logger.info(f'Logging image samples from train [num_samples={self.num_samples}]')
        ds = trainer.train_dataloader.dataset
        self.visualize(trainer=trainer, name='train', dataset=ds)


class DINOImageSamples(Callback):

    def __init__(self, num_samples: int = 5, seed: int = 0, padding: int = 2):
        self.num_samples = num_samples
        self.padding = padding
        self.rng = np.random.default_rng(seed=seed)

    def on_train_start(self, trainer, pl_module):
        if not trainer.fast_dev_run:
            return

        logger.info(f'Logging DINO image samples [num_samples={self.num_samples}]')

        ds = trainer.train_dataloader.dataset

        # random indices
        indices = np.arange(len(ds))
        self.rng.shuffle(indices)
        indices = indices[:self.num_samples]

        items = [ds[i] for i in indices]

        num_global_views = len(items[0]['global_views'])
        num_local_views = len(items[0]['local_views'])

        global_views = []
        local_views = []
        for item in items:
            global_views.extend([view['image'] for view in item['global_views']])
            local_views.extend([view['image'] for view in item['local_views']])

        global_grid = make_grid(global_views, nrow=num_global_views, normalize=True, padding=self.padding)
        local_grid = make_grid(local_views, nrow=num_local_views, normalize=True, padding=self.padding)

        trainer.logger.log_image(key="image_samples/global_views",
                                 images=[global_grid],
                                 caption=["global_views"])
        trainer.logger.log_image(key="image_samples/local_views",
                                 images=[local_grid],
                                 caption=["local_views"])
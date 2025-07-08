from lightning.pytorch.callbacks import Callback
import numpy as np
from torchvision.utils import make_grid
from loguru import logger

class ImageSamples(Callback):

    def __init__(self, from_train: bool = True, from_val: bool = True,
                 num_samples: int = 5, seed: int = 0, padding: int = 2):

        self.from_train = from_train
        self.from_val = from_val
        self.num_samples = num_samples
        self.padding = padding
        self.rng = np.random.default_rng(seed=seed)

    def on_train_start(self, trainer, pl_module):
        if trainer.sanity_checking:
            return

        logger.info(f'Logging image samples [num_samples={self.num_samples}]')

        ds_train = trainer.train_dataloader.dataset
        ds_val = trainer.val_dataloaders.dataset

        dataloaders = []
        if self.from_train:
            dataloaders.append(('train', ds_train))
        if self.from_val:
            dataloaders.append(('val', ds_val))

        for name, ds in dataloaders:
            # random indices
            idc = np.arange(len(ds_train))
            self.rng.shuffle(idc)
            idc = idc[:self.num_samples]

            items = [ds[i] for i in idc]
            images = [i['image'] for i in items]

            grid = make_grid(images, normalize=True, padding=self.padding)

            trainer.logger.log_image(key=f"image_samples/{name}", images=[grid], epoch=trainer.current_epoch)

class DINOImageSamples(Callback):

    def __init__(self, num_samples: int = 5, seed: int = 0, padding: int = 2):
        self.num_samples = num_samples
        self.padding = padding
        self.rng = np.random.default_rng(seed=seed)

    def on_train_start(self, trainer, pl_module):
        if trainer.sanity_checking:
            return

        logger.info(f'Logging image samples [num_samples={self.num_samples}]')

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

        trainer.logger.log_image(key="image_samples",
                                 images=[global_grid],
                                 caption=["global_views"])
        trainer.logger.log_image(key="image_samples",
                                 images=[local_grid],
                                 caption=["local_views"])
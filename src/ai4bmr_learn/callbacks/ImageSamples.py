from lightning.pytorch.callbacks import Callback
import numpy as np
from torchvision.utils import make_grid
from loguru import logger

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

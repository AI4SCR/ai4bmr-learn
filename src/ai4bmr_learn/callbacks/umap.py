import numpy as np
import torch
import wandb
from glom import glom
from lightning.pytorch.callbacks import Callback
from loguru import logger
from tqdm import tqdm

from ai4bmr_learn.callbacks.cache import ValidationCache
from ai4bmr_learn.plotting.umap import plot_umap
from ai4bmr_learn.utils.device import batch_to_device


class UMAP(Callback):

    def __init__(self,
                 before_train: bool = False,
                 after_train: bool = False,
                 every_num_epochs=1,
                 num_samples: int | None = None,
                 label_key: str | None = None,
                 value_key: str | None = None,
                 n_neighbors: int = 15,
                 min_dist: float = 0.3,
                 metric: str = 'euclidean',
                 engine: str = 'umap-learn',
                 umap_kwargs: dict | None = None,
                 show_legend: bool = True,
                 **kwargs):

        self.before_train = before_train
        self.after_train = after_train
        self.every_num_epochs = every_num_epochs
        self.num_samples = num_samples
        self.label_key = label_key
        self.value_key = value_key
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.metric = metric
        self.engine = engine
        self.umap_kwargs = umap_kwargs
        self.show_legend = show_legend
        self.kwargs = kwargs

        self.embeddings = self.labels = self.values = None

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return

        cache, = list(filter(lambda x: isinstance(x, ValidationCache), trainer.callbacks))
        x, values, labels = self.get_data(cache=cache)
        logger.info(f'UMAP [num_samples={x.shape[0]}, epoch={trainer.current_epoch}]')
        self.run_umap(x=x, values=values, labels=labels, trainer=trainer)

    def run_umap(self, x, values, labels, trainer):

        ax = plot_umap(data=x, labels=labels, values=values, n_neighbors=self.n_neighbors,
                       min_dist=self.min_dist, metric=self.metric, engine=self.engine, show_legend=self.show_legend,
                       umap_kwargs=self.umap_kwargs)

        image = wandb.Image(ax)
        if not trainer.fast_dev_run:
            trainer.logger.experiment.log({"umap/": image,
                                           "epoch": trainer.current_epoch, "trainer/global_step": trainer.global_step})

    def get_data(self, cache) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = torch.vstack([i['embedding'] for i in cache.outputs]).detach().cpu().numpy()
        num_samples = self.num_samples or x.shape[0]

        values = []
        labels = []
        for batch in cache.outputs:
            if self.value_key is not None:
                val = glom(batch, self.value_key)
                val = val.detach().cpu().numpy() if isinstance(val, torch.Tensor) else val
                values.append(val)

            if self.label_key is not None:
                label = glom(batch, self.label_key)
                label = label.detach().cpu().numpy() if isinstance(label, torch.Tensor) else label
                labels.append(label)

        x = x[:num_samples]
        values = np.concatenate(values)[:num_samples] if values else None
        labels = np.concatenate(labels)[:num_samples] if labels else None

        return x, values, labels

    # def get_outputs(self, trainer, pl_module):
    #     dl = trainer.val_dataloaders
    #     assert pl_module.device.type == 'cuda'
    #     with torch.no_grad():
    #         for batch_idx, batch in tqdm(enumerate(dl), desc='collecting outputs'):
    #             batch = batch_to_device(batch, pl_module.device)
    #             outputs = pl_module.validation_step(batch=batch, batch_idx=batch_idx)
    #             is_accumulated = self.accumulate(outputs)
    #
    #             if is_accumulated or trainer.fast_dev_run:
    #                 break
    #
    #     return outputs, batch_idx
    #
    # def on_train_start(self, trainer, pl_module):
    #     if self.before_train and not trainer.fast_dev_run:
    #         logger.info(f'UMAP on_train_start')
    #         outputs, batch_idx = self.get_outputs(trainer, pl_module)
    #         self.accumulate(outputs)
    #         self.run_umap(trainer=trainer)
    #         self.reset()
    #
    # def on_train_end(self, trainer, pl_module):
    #     if self.after_train and not trainer.fast_dev_run:
    #         logger.info(f'UMAP on_train_end')
    #         outputs, batch_idx = self.get_outputs(trainer, pl_module)
    #         self.accumulate(outputs)
    #         self.run_umap(trainer=trainer)
    #         self.reset()
    #
    # def should_run(self, trainer, force):
    #     if trainer.sanity_checking:
    #         return False
    #
    #     if trainer.fast_dev_run:
    #         return False
    #
    #     if force:
    #         return True
    #
    #     if trainer.current_epoch == 0 and self.before_train:
    #         return False
    #
    #     if trainer.current_epoch % self.every_num_epochs == 0:
    #         return True
    #
    #     return False
    #
    # def reset(self):
    #     self.embeddings = self.targets = None

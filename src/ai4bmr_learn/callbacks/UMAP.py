import numpy as np
import torch
import wandb
from glom import glom
from lightning.pytorch.callbacks import Callback
from loguru import logger
from tqdm import tqdm

from ai4bmr_learn.plotting.umap import plot_umap
from ai4bmr_learn.utils.device import batch_to_device


class UMAP(Callback):

    def __init__(self,
                 run_before_train: bool = True,
                 run_after_train: bool = True,
                 run_every_num_epochs=1,
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

        self.run_before_train = run_before_train
        self.run_after_train = run_after_train
        self.run_every_num_epochs = run_every_num_epochs
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

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0,
                                force: bool = False):

        if not self.should_run(trainer=trainer, force=force):
            return

        is_accumulated = self.accumulate(outputs)
        # note: trainer.is_last_batch is not always reset for some reason
        is_last_batch = len(trainer.val_dataloaders) - 1 == batch_idx

        if is_last_batch:
            logger.info(
                f'UMAP [num_samples={len(self.embeddings)}, epoch={trainer.current_epoch}, batch_idx={batch_idx}]')
            self.run_umap(trainer=trainer)
            self.reset()

    def run_umap(self, trainer):
        if self.num_samples is not None:
            self.embeddings = self.embeddings[:self.num_samples]

        # TODO: use .numpy() instead of using lists
        labels = np.array(self.labels) if self.label_key is not None else None
        values = np.array(self.values) if self.value_key is not None else None
        ax = plot_umap(data=self.embeddings, labels=labels, values=values, n_neighbors=self.n_neighbors,
                       min_dist=self.min_dist, metric=self.metric, engine=self.engine, show_legend=self.show_legend,
                       umap_kwargs=self.umap_kwargs)

        image = wandb.Image(ax)
        trainer.logger.experiment.log({"umap/": image,
                                       "epoch": trainer.current_epoch, "trainer/global_step": trainer.global_step})
        self.embeddings = self.labels = self.values = None

    def accumulate(self, outputs) -> bool:
        accumulate = (self.num_samples is None) or self.embeddings is None or (len(self.embeddings) < self.num_samples)

        if accumulate and self.embeddings is None:
            self.embeddings = outputs['embedding']
            if self.label_key is not None:
                labels = glom(outputs, self.label_key)
                labels = labels.tolist() if isinstance(labels, torch.Tensor) else labels
                self.labels = labels
            if self.value_key is not None:
                values = glom(outputs, self.label_key)
                values = values.tolist() if isinstance(values, torch.Tensor) else values
                self.values = values
        elif accumulate:
            self.embeddings = torch.vstack((self.embeddings, outputs['embedding']))
            if self.label_key is not None:
                labels = glom(outputs, self.label_key)
                labels = labels.tolist() if isinstance(labels, torch.Tensor) else labels
                self.labels.extend(labels)
            if self.value_key is not None:
                values = glom(outputs, self.label_key)
                values = values.tolist() if isinstance(values, torch.Tensor) else values
                self.values.extend(values)
        else:
            return True
        return False

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

    def on_train_start(self, trainer, pl_module):
        if self.run_before_train and not trainer.fast_dev_run:
            logger.info(f'UMAP on_train_start')
            outputs, batch_idx = self.get_outputs(trainer, pl_module)
            self.accumulate(outputs)
            self.run_umap(trainer=trainer)
            self.reset()

    def on_train_end(self, trainer, pl_module):
        if self.run_after_train and not trainer.fast_dev_run:
            logger.info(f'UMAP on_train_end')
            outputs, batch_idx = self.get_outputs(trainer, pl_module)
            self.accumulate(outputs)
            self.run_umap(trainer=trainer)
            self.reset()

    def should_run(self, trainer, force):
        if trainer.sanity_checking:
            return False

        if trainer.fast_dev_run:
            return False

        if force:
            return True

        if trainer.current_epoch == 0 and self.run_before_train:
            return False

        if trainer.current_epoch % self.run_every_num_epochs == 0:
            return True

        return False

    def reset(self):
        self.embeddings = self.targets = None

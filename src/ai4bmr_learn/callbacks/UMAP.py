from lightning.pytorch.callbacks import Callback
from loguru import logger
from ai4bmr_learn.plotting.umap import plot_umap
import wandb
import torch
from ai4bmr_learn.utils.device import batch_to_device
import numpy as np
from torch.utils.data import DataLoader

class UMAP(Callback):

    def __init__(self,
                 log_before_train: bool = True,
                 log_every_num_epochs=10,
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

        self.log_before_train = log_before_train,
        self.log_every_num_epochs = log_every_num_epochs
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

    def shared_step(self, trainer, pl_module, force: bool = False):
        if trainer.sanity_checking:
            return

        dl = trainer.val_dataloaders
        assert pl_module.device.type == 'cuda'
        with torch.no_grad():
            for batch_idx, batch in enumerate(dl):
                batch = batch_to_device(batch, pl_module.device)
                outputs = pl_module.validation_step(batch=batch, batch_idx=batch_idx)
                is_last_batch = batch_idx == len(dl) - 1
                completed = self.on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx,
                                                         is_last_batch=is_last_batch, force=force)

                if completed:
                    break

    def on_train_start(self, trainer, pl_module):
        self.shared_step(trainer, pl_module, force=True)

    def on_train_end(self, trainer, pl_module):
        self.shared_step(trainer, pl_module, force=True)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0,
                                is_last_batch: bool | None = None, force: bool = False):
        if trainer.sanity_checking:
            return

        if (trainer.current_epoch % self.log_every_num_epochs > 0 or trainer.current_epoch == 0) and not force:
            return

        is_last_batch = is_last_batch or len(trainer.val_dataloaders) - 1 == batch_idx  # note: trainer.is_last_batch is not reset for some reason
        accumulate = (self.num_samples is None) or (len(self.embeddings) < self.num_samples)
        if accumulate and self.embeddings is None:
            self.embeddings = outputs['embedding']
            if self.label_key is not None:
                self.labels = outputs[self.label_key].tolist()
            if self.value_key is not None:
                self.values = outputs[self.value_key].tolist()
        else:
            self.embeddings = torch.vstack((self.embeddings, outputs['embedding']))
            if self.label_key is not None:
                self.labels.extend(outputs[self.label_key].tolist())
            if self.value_key is not None:
                self.values.extend(outputs[self.value_key].tolist())

        if is_last_batch:
            logger.info(f'Computing UMAP [num_samples={len(self.embeddings)}, epoch={trainer.current_epoch}, batch_idx={batch_idx}]')

            if self.num_samples is not None:
                self.embeddings = self.embeddings[:self.num_samples]

            # TODO: use .numpy() instead of using lists
            labels = np.array(self.labels) if self.label_key is not None else None
            values = np.array(self.values) if self.value_key is not None else None
            ax = plot_umap(data=self.embeddings, labels=labels, values=values, n_neighbors=self.n_neighbors,
                      min_dist=self.min_dist, metric=self.metric, engine=self.engine, show_legend=self.show_legend,
                      umap_kwargs=self.umap_kwargs)

            image = wandb.Image(ax)
            trainer.logger.experiment.log({"umap": image, "epoch": trainer.current_epoch})
            self.embeddings = self.labels = self.values = None

            return True
        return False


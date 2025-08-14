from typing import Any

import lightning as L
import torch
import torch.nn as nn
from glom import glom

class Embeddings(L.LightningModule):

    def __init__(self,
                 backbone: nn.Module,
                 save_key: str,
                 batch_key: str | None = None,
                 drop_keys: list[str] | None = None,
                 pooling: str | None = None,
                 as_kwargs: bool = False
                 ):
        super().__init__()

        self.backbone = backbone
        self.save_key = save_key

        self.batch_key = batch_key
        self.drop_keys = drop_keys
        self.pooling = pooling
        self.as_kwargs = as_kwargs

        self.save_hyperparameters(ignore=[backbone])

    def predict_step(self, batch: dict) -> Any:
        data = glom(batch, self.batch_key) if self.batch_key is not None else batch
        out = self.backbone(**data) if self.as_kwargs else self.backbone(data)

        for key in self.drop_keys:
            del data[key]

        batch[self.save_key] = out

        return batch

    def pool(self, x):
        if self.pooling is None:
            return x
        elif self.pooling == 'cls':
            return x[:, 0]
        elif self.pooling == 'flatten':
            return x.flatten(start_dim=1)
        else:
            raise NotImplementedError(f'{self.pooling} is not implemented.')
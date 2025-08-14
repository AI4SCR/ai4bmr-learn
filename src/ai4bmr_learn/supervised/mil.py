import lightning as L
import torch.optim as optim
import torch.nn as nn
from ai4bmr_learn.metrics.classification import get_metric_collection
from glom import glom
import torch
import numpy as np

class MIL(L.LightningModule):
    def __init__(self,
                 backbone: nn.Module,
                 head: nn.Module,
                 num_classes: int,
                 lr: float = 1e-3,
                 lr_backbone: float = 1e-4,
                 weight_decay: float = 0.01,
                 freeze_backbone: bool = False,
                 pooling: str | None = None,
                 batch_key: str | None = 'image',
                 data_keys: list[str] | None = None,
                 target_key: str = 'target',
                 attention_key: str = 'attention',
                 as_kwargs: bool = False,
                 ):

        super().__init__()

        # MODULES
        self.backbone = backbone
        self.head = head

        # LOSS
        self.criterion = nn.CrossEntropyLoss()

        # METRICS
        self.num_classes = num_classes
        metrics = get_metric_collection(num_classes=num_classes)
        self.train_metrics = metrics.clone(prefix="train/")
        self.valid_metrics = metrics.clone(prefix="val/")
        self.test_metrics = metrics.clone(prefix="test/")

        # PARAMS
        self.lr = lr
        self.lr_backbone = lr_backbone
        self.weight_decay = weight_decay
        self.freeze_backbone = freeze_backbone
        self.pooling = pooling

        # DATA
        self.batch_key = batch_key
        self.data_keys = data_keys
        if self.data_keys is None:
            self.data_keys = [batch_key]
        self.target_key = target_key
        self.attention_key = attention_key
        self.as_kwargs = as_kwargs

    def shared_step_for_list(self, batch: list, batch_idx: int | None = None):

        zs = []
        targets = []
        # NOTE: we need to feed the bags separately to the backbone, since they are designed for [B, D] inputs
        for i, bag in enumerate(batch):

            target = glom(bag, self.target_key)
            target = torch.unique(target).item()
            targets.append(target)

            data = glom(bag, self.batch_key) if self.batch_key else bag

            z = self.backbone(**data) if self.as_kwargs else self.backbone(data)
            z = self.pool(z)
            zs.append(z)

        zs = torch.stack(zs)  # B, M, R
        targets = torch.tensor(targets)
        assert zs.shape[0] == targets.shape[0]

        logits = self.head(zs)
        loss = self.criterion(logits, targets)

        return zs, logits, targets, loss

    def shared_step_for_dict(self, batch, batch_idx: int | None = None):
        bags = glom(batch, self.batch_key)
        attentions = glom(batch, self.attention_key, default=None)
        assert isinstance(bags, torch.Tensor)
        assert attentions is None or isinstance(attentions, torch.Tensor)
        if attentions is not None and not isinstance(bags, torch.Tensor):
            raise NotImplementedError('Bags must be pure tensors if you want to use padding. Use `collate_fn=list` and pad=False instead.')

        targets = glom(batch, self.target_key)
        targets = torch.unique(targets, dim=1).squeeze()  # NOTE: we expect [B, M], #TODO: discuss this choice
        assert targets.ndim == 1

        B, M, *D = bags.shape
        zs = []
        # NOTE: we need to feed the bags separately to the backbone, since they are designed for [B, D] inputs
        for i, bag in enumerate(bags):

            if attentions is not None:
                attention = attentions[i]
                bag = bag[attention]

            z = self.backbone(bag)
            z = self.pool(z)
            zs.append(z)

        zs = torch.stack(zs)  # B, M, R
        assert zs.shape[0] == B and zs.shape[1] == M

        logits = self.head(zs)
        loss = self.criterion(logits, targets)

        return zs, logits, targets, loss

    def shared_step(self, batch, batch_idx: int | None = None):
        if isinstance(batch, list):
            return self.shared_step_for_list(batch, batch_idx=batch_idx)
        elif isinstance(batch, dict):
            return self.shared_step_for_dict(batch, batch_idx=batch_idx)

    def training_step(self, batch, batch_idx):
        z, logits, targets, loss = self.shared_step(batch)
        batch_size = targets.shape[0]

        # metrics
        self.train_metrics(logits, targets)
        self.log_dict(self.train_metrics, batch_size=batch_size)

        # loss
        self.log("train/loss", loss, batch_size=batch_size)


        if isinstance(batch, list):
            return loss
        else:
            batch['loss'] = loss
            batch['embedding'] = z.detach().cpu()
            return batch

    def validation_step(self, batch, batch_idx):
        z, logits, targets, loss = self.shared_step(batch)
        batch_size = targets.shape[0]

        # metrics
        self.valid_metrics(logits, targets)
        self.log_dict(self.valid_metrics, batch_size=batch_size)

        # loss
        self.log("val/loss", loss, batch_size=batch_size)

        if isinstance(batch, list):
            return loss
        else:
            batch['loss'] = loss
            batch['embedding'] = z.detach().cpu()
            return batch

    def test_step(self, batch, batch_idx):
        z, logits, targets, loss = self.shared_step(batch, batch_idx)
        batch_size = targets.shape[0]

        # metrics
        self.test_metrics(logits, targets)
        self.log_dict(self.test_metrics, batch_size=batch_size)

        # loss
        self.log("test/loss", loss, batch_size=batch_size)

        batch['loss'] = loss
        batch['embedding'] = z.detach().cpu()
        return batch

    def predict_step(self, batch, batch_idx):
        data = glom(batch, self.batch_key)

        z = []
        for bag in data:
            z.append(self.pool(self.backbone(bag)))
        z = torch.stack(z)

        logits = self.head(z)
        preds = logits.argmax(dim=1)

        batch["prediction"] = preds.detach().cpu()
        batch["logits"] = logits.detach().cpu()
        batch['embedding'] = z.detach().cpu()
        return batch

    def configure_optimizers(self):
        optimizer = optim.Adam([
            {'params': self.head.parameters(), 'lr': self.lr},
            {'params': filter(lambda p: p.requires_grad, self.backbone.parameters()), 'lr': self.lr_backbone}
        ], weight_decay=self.weight_decay)

        return optimizer

    def pool(self, x):
        if self.pooling is None:
            return x
        elif self.pooling == 'cls':
            return x[:, 0]
        elif self.pooling == 'flatten':
            return x.flatten(start_dim=1)
        else:
            raise NotImplementedError(f'{self.pooling} is not implemented.')
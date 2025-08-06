import lightning as L
import torch.optim as optim
import torch.nn as nn
from ai4bmr_learn.metrics.classification import get_metric_collection
from glom import glom
import torch

class MIL(L.LightningModule):
    def __init__(self,
                 backbone: nn.Module,
                 head: nn.Module,
                 num_classes: int,
                 lr: float = 1e-3,
                 lr_backbone: float = 1e-4,
                 weight_decay: float = 0.01,
                 freeze_backbone: bool = False,
                 pooling: str = 'flatten',
                 batch_key: str | None = 'image',
                 target_key: str = 'label',
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
        self.target_key = target_key

    def shared_step(self, batch):
        data = glom(batch, self.batch_key)
        targets = glom(batch, self.target_key)

        B, M, *D = data.shape
        # data = data.flatten(end_dim=1)
        logits = []
        for bag in data:
            z = self.backbone(bag)
            z = self.pool(z)
            z = self.head(z)
            logits.append(z)
        logits = torch.cat(logits)

        loss = self.criterion(logits, targets)

        return targets, logits, loss

    def training_step(self, batch, batch_idx):
        targets, logits, loss = self.shared_step(batch)  # data: B, M, D

    def validation_step(self, batch, batch_idx):
        pass

    def pool(self, x):
        if self.pooling is None:
            return x
        elif self.pooling == 'cls':
            return x[:, 0]
        elif self.pooling == 'flatten':
            return x.flatten(start_dim=1)
        else:
            raise NotImplementedError(f'{self.pooling} is not implemented.')
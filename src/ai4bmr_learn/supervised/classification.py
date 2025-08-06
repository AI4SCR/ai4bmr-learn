import lightning as L
import torch.optim as optim
import torch.nn as nn
from ai4bmr_learn.metrics.classification import get_metric_collection
from glom import glom

class Classifier(L.LightningModule):
    def __init__(self,
                 backbone: nn.Module,
                 input_dim: int,
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

        self.save_hyperparameters(ignore=["backbone"])

        self.backbone = backbone
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.head = nn.Linear(input_dim, num_classes)
        self.pooling = pooling
        self.batch_key = batch_key
        self.target_key = target_key

        self.lr = lr
        self.lr_backbone = lr_backbone
        self.weight_decay = weight_decay

        self.criterion = nn.CrossEntropyLoss()

        # METRICS
        # task = "multiclass" if num_classes > 2 else "binary"
        metrics = get_metric_collection(num_classes=num_classes)
        self.train_metrics = metrics.clone(prefix="train/")
        self.valid_metrics = metrics.clone(prefix="val/")
        self.test_metrics = metrics.clone(prefix="test/")

    def _shared_step(self, batch, batch_idx):
        # print(batch.keys())
        data = glom(batch, self.batch_key) if self.batch_key is not None else batch
        # data.to(self.device)

        y = self.backbone(data)
        y = self.pool(y)
        logits = self.head(y)
        targets = glom(batch, self.target_key).long()

        loss = self.criterion(logits, targets)

        return y, logits, targets, loss

    def training_step(self, batch, batch_idx):
        y, logits, targets, loss = self._shared_step(batch, batch_idx)
        batch_size = targets.shape[0]

        # metrics
        self.train_metrics(logits, targets)
        self.log_dict(self.train_metrics, batch_size=batch_size)

        # loss
        self.log("train_loss", loss.item(), batch_size=batch_size)

        batch['loss'] = loss
        batch['embedding'] = y.detach().cpu()
        return batch

    def validation_step(self, batch, batch_idx):
        y, logits, targets, loss = self._shared_step(batch, batch_idx)
        batch_size = targets.shape[0]

        # metrics
        self.valid_metrics(logits, targets)
        self.log_dict(self.valid_metrics, batch_size=batch_size)

        # loss
        self.log("val_loss", loss.item(), batch_size=batch_size)

        batch['loss'] = loss
        batch['embedding'] = y.detach().cpu()
        return batch

    def test_step(self, batch, batch_idx):
        y, logits, targets, loss = self._shared_step(batch, batch_idx)
        batch_size = targets.shape[0]

        # metrics
        self.test_metrics(logits, targets)
        self.log_dict(self.test_metrics, batch_size=batch_size)

        # loss
        self.log("test_loss", loss.item(), batch_size=batch_size)

        batch['loss'] = loss
        batch['embedding'] = y.detach().cpu()
        return batch

    def predict_step(self, batch, batch_idx):
        images = batch['image']
        y = self.backbone(images)
        y = self.pool(y)
        logits = self.head(y)
        preds = logits.argmax(dim=1)

        batch["prediction"] = preds.detach().cpu()
        batch["logits"] = logits.detach().cpu()
        batch['embedding'] = y.detach().cpu()
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
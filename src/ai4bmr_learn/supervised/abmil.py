import lightning as L
import torch.optim as optim
import torch.nn as nn
from ai4bmr_learn.metrics.classification import get_metric_collection
from glom import glom
from collections import Counter
from ai4bmr_learn.utils.pooling import pool
import torch

class Aggregator(nn.Module):
    """Attention-Based MIL for precomputed embeddings."""
    def __init__(self, input_dim: int, use_projection: bool = False):
        super().__init__()
        self.use_projection = use_projection
        self.proj = nn.Linear(input_dim, input_dim) if use_projection else nn.Identity()
        self.act = nn.Tanh() if use_projection else nn.Identity()
        self.attention = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor):
        attn = self.act(self.proj(x))
        attn = self.attention(attn)
        attn = torch.softmax(attn, dim=1)
        z = torch.sum(attn * x, dim=1)
        return z, attn.squeeze(-1)

class ABMIL(L.LightningModule):
    def __init__(self,
                 # backbone: nn.Module,
                 input_dim: int,
                 num_classes: int,
                 use_projection: bool = False,
                 lr_head: float = 1e-3,
                 lr_mil: float = 1e-4,
                 weight_decay: float = 0.01,
                 # freeze_backbone: bool = False,
                 # pooling: str | None = None,
                 batch_key: str | None = 'image',
                 target_key: str = 'label',
                 ):
        super().__init__()

        self.save_hyperparameters(ignore=["mil"])
        self.mil = Aggregator(input_dim, use_projection=use_projection)

        # self.backbone = backbone
        # if freeze_backbone:
        #     for param in self.backbone.parameters():
        #         param.requires_grad = False

        self.head = nn.Linear(input_dim, num_classes)
        # self.pooling = pooling
        self.batch_key = batch_key
        self.target_key = target_key

        self.lr_head = lr_head
        self.lr_mil = lr_mil
        self.weight_decay = weight_decay

        self.criterion = nn.CrossEntropyLoss()

        # METRICS
        self.num_classes = num_classes
        metrics = get_metric_collection(num_classes=num_classes)
        self.train_metrics = metrics.clone(prefix="train/")
        self.valid_metrics = metrics.clone(prefix="val/")
        self.test_metrics = metrics.clone(prefix="test/")

        # STATS
        self.train_stats = {'train/num_samples': 0, 'class_cnt': Counter()}
        self.val_stats = {'val/num_samples': 0, 'class_cnt': Counter()}

    def shared_step(self, batch, batch_idx):
        data = glom(batch, self.batch_key) if self.batch_key is not None else batch

        # y = self.backbone(data)
        # y = pool(y, strategy=self.pooling)

        z, attn = self.mil(data)
        logits = self.head(z)
        targets = glom(batch, self.target_key).long()

        loss = self.criterion(logits, targets)

        return z, logits, targets, loss

    def on_train_epoch_start(self):
        self.train_stats['train/num_samples'] = 0
        self.train_stats['class_cnt'] = Counter()

    def on_train_epoch_end(self):
        class_cnt = self.train_stats.pop('class_cnt')
        class_counts = {f'train/class_{k}': v for k, v in class_cnt.items()}
        self.train_stats.update(class_counts)

        self.log_dict(self.train_stats)

        self.train_stats['train/num_samples'] = 0
        self.train_stats['class_cnt'] = Counter()

    def training_step(self, batch, batch_idx):
        z, logits, targets, loss = self.shared_step(batch, batch_idx)
        batch_size = targets.shape[0]

        # metrics
        # logits = logits.argmax(dim=1) if self.num_classes == 2 else logits
        logits = logits[:, 1] if self.num_classes == 2 else logits
        self.train_metrics(logits, targets)
        self.log_dict(self.train_metrics, on_step=True, on_epoch=True, batch_size=batch_size)

        # loss
        self.log("loss/train", loss, on_step=True, on_epoch=True, batch_size=batch_size)

        # stats
        self.train_stats['train/num_samples'] += batch_size
        self.train_stats['class_cnt'].update(targets.tolist())

        batch['loss'] = loss
        batch['repr'] = z.detach().cpu()
        return batch

    def on_val_epoch_start(self):
        self.val_stats['val/num_samples'] = 0
        self.val_stats['class_cnt'] = Counter()

    def on_val_epoch_end(self):
        class_cnt = self.val_stats.pop('class_cnt')
        class_counts = {f'val/class_{k}': v for k, v in class_cnt.items()}
        self.val_stats.update(class_counts)

        self.log_dict(self.val_stats)

        self.val_stats['val/num_samples'] = 0
        self.val_stats['class_cnt'] = Counter()

    def validation_step(self, batch, batch_idx):
        z, logits, targets, loss = self.shared_step(batch, batch_idx)
        batch_size = targets.shape[0]

        # metrics
        # logits = logits.argmax(dim=1) if self.num_classes == 2 else logits
        logits = logits[:, 1] if self.num_classes == 2 else logits
        self.valid_metrics(logits, targets)
        self.log_dict(self.valid_metrics, on_step=False, on_epoch=True, batch_size=batch_size)

        # loss
        self.log("loss/val", loss, batch_size=batch_size)

        # stats
        self.val_stats['val/num_samples'] += batch_size
        self.val_stats['class_cnt'].update(targets.tolist())

        batch['loss'] = loss
        batch['repr'] = z.detach().cpu()
        return batch

    def test_step(self, batch, batch_idx):
        z, logits, targets, loss = self.shared_step(batch, batch_idx)
        batch_size = targets.shape[0]

        # metrics
        # logits = logits.argmax(dim=1) if self.num_classes == 2 else logits
        logits = logits[:, 1] if self.num_classes == 2 else logits
        self.test_metrics(logits, targets)
        self.log_dict(self.test_metrics, batch_size=batch_size)

        # loss
        self.log("loss/test", loss, batch_size=batch_size)

        batch['loss'] = loss
        batch['repr'] = z.detach().cpu()
        return batch

    def configure_optimizers(self):
        optimizer = optim.Adam([
            {'params': self.head.parameters(), 'lr': self.lr_head},
            {'params': filter(lambda p: p.requires_grad, self.mil.parameters()), 'lr': self.lr_mil}
        ], weight_decay=self.weight_decay)

        return optimizer

import lightning as L
import torch.optim as optim
from ai4bmr_learn.metrics.classification import get_metric_collection
from torchmetrics.classification import ConfusionMatrix
from glom import glom
from collections import Counter
from ai4bmr_learn.utils.pooling import pool
import torch
from torch import nn

import torch
import torch.nn as nn


class MIL(nn.Module):
    def __init__(
            self,
            input_dim: int,
            activ_dim: int = 0,  # if >0, project to this dim before attention
            bag_norm: bool = False,  # z-score across instances in a bag
            instance_norm: bool = False,  # LayerNorm per instance (feature-wise)
            input_dropout: float = 0.0,
            activ_dropout: float = 0.0,
            attn_dropout: float = 0.0,
    ):
        super().__init__()

        # If we want activation dropout, we must actually have an activation layer
        assert not (activ_dropout > 0.0 and activ_dim <= 0), \
            "activ_dropout > 0 requires activ_dim > 0"

        self.activ_dim = activ_dim
        self.attn_dim = activ_dim if activ_dim > 0 else input_dim
        self.output_dim = self.attn_dim  # for compatibility with ABMIL

        self.bag_norm = bag_norm
        self.instance_norm = instance_norm

        self.input_drop = nn.Dropout(input_dropout) if input_dropout > 0 else nn.Identity()
        self.activ_drop = nn.Dropout(activ_dropout) if activ_dropout > 0 else nn.Identity()
        self.attn_drop = nn.Dropout(attn_dropout) if attn_dropout > 0 else nn.Identity()

        if activ_dim > 0:
            self.activate = nn.Sequential(
                nn.Linear(input_dim, activ_dim),
                nn.Tanh(),
                self.activ_drop,
            )
        else:
            self.activate = nn.Identity()

        # LayerNorm over features of each instance (last dim)
        self.layer_norm = nn.LayerNorm(input_dim)

        # Attention over the (possibly activated) features
        self.attention = nn.Linear(self.attn_dim, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x: [B, N, D]
        Returns:
            z:    [B, D_or_activ]  aggregated embeddings (matches attn input dim)
            attn: [B, N]           attention weights
        """
        if self.bag_norm:
            # Z-score within each bag across instances
            mean = x.mean(dim=1, keepdim=True)
            std = x.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
            x = (x - mean) / std

        if self.instance_norm:
            x = self.layer_norm(x)

        x = self.input_drop(x)
        x = self.activate(x)  # shape: [B, N, attn_dim]

        logits = self.attention(x)  # [B, N, 1]
        attn = torch.softmax(logits, dim=1)  # [B, N, 1]
        attn = self.attn_drop(attn)  # dropout on probs

        z = torch.sum(attn * x, dim=1)  # [B, attn_dim]
        return z, attn.squeeze(-1)  # [B, attn_dim], [B, N]


class MILTrainer(L.LightningModule):
    def __init__(self,
                 mil: nn.Module,
                 num_classes: int,
                 lr_head: float = 1e-3,
                 lr_mil: float = 1e-4,
                 weight_decay: float = 0.01,
                 # freeze_backbone: bool = False,
                 # pooling: str | None = None,
                 batch_key: str | None = 'image',
                 target_key: str = 'label',
                 weight: torch.Tensor | None = None
                 ):
        super().__init__()

        self.save_hyperparameters(ignore=["mil"])
        self.mil = mil

        # self.backbone = backbone
        # if freeze_backbone:
        #     for param in self.backbone.parameters():
        #         param.requires_grad = False

        self.head = nn.Linear(mil.output_dim, num_classes)
        # self.pooling = pooling
        self.batch_key = batch_key
        self.target_key = target_key

        self.lr_head = lr_head
        self.lr_mil = lr_mil
        self.weight_decay = weight_decay

        self.criterion = nn.CrossEntropyLoss(weight=weight)

        # METRICS
        self.num_classes = num_classes
        task = "binary" if num_classes == 2 else "multiclass"

        metrics = get_metric_collection(num_classes=num_classes)
        self.train_metrics = metrics.clone(prefix="train/")
        self.valid_metrics = metrics.clone(prefix="val/")
        self.test_metrics = metrics.clone(prefix="test/")

        normalize = None
        self.cm_train = ConfusionMatrix(task=task, normalize=normalize, num_classes=num_classes)
        self.cm_val = ConfusionMatrix(task=task, normalize=normalize, num_classes=num_classes)
        self.cm_test = ConfusionMatrix(task=task, normalize=normalize, num_classes=num_classes)

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

        fig, ax = self.cm_train.plot()
        self.logger.experiment.log({'train/confusion_matrix': fig})

        self.train_stats['train/num_samples'] = 0
        self.train_stats['class_cnt'] = Counter()

    def training_step(self, batch, batch_idx):
        z, logits, targets, loss = self.shared_step(batch, batch_idx)
        batch_size = targets.shape[0]

        # metrics
        # logits = logits.argmax(dim=1) if self.num_classes == 2 else logits
        preds = torch.argmax(logits, dim=1).long()
        # logits = logits[:, 1] if self.num_classes == 2 else logits
        self.train_metrics(logits, targets)
        self.log_dict(self.train_metrics, on_step=True, on_epoch=True, batch_size=batch_size)

        self.cm_train(preds, targets)

        # loss
        self.log("loss/train", loss, on_step=True, on_epoch=True, batch_size=batch_size)

        # stats
        self.train_stats['train/num_samples'] += batch_size
        self.train_stats['class_cnt'].update(targets.tolist())

        batch['loss'] = loss
        batch['repr'] = z.detach().cpu()
        return batch

    def on_validation_epoch_start(self):
        self.val_stats['val/num_samples'] = 0
        self.val_stats['class_cnt'] = Counter()

    def on_validation_epoch_end(self):
        class_cnt = self.val_stats.pop('class_cnt')
        class_counts = {f'val/class_{k}': v for k, v in class_cnt.items()}
        self.val_stats.update(class_counts)

        self.log_dict(self.val_stats)

        fig, ax = self.cm_val.plot()
        self.logger.experiment.log({'val/confusion_matrix': fig})

        self.val_stats['val/num_samples'] = 0
        self.val_stats['class_cnt'] = Counter()

    def validation_step(self, batch, batch_idx):
        z, logits, targets, loss = self.shared_step(batch, batch_idx)
        batch_size = targets.shape[0]

        # metrics
        # logits = logits.argmax(dim=1) if self.num_classes == 2 else logits
        preds = torch.argmax(logits, dim=1).long()
        # logits = logits[:, 1] if self.num_classes == 2 else logits
        self.valid_metrics(logits, targets)
        self.log_dict(self.valid_metrics, on_step=False, on_epoch=True, batch_size=batch_size)

        self.cm_val(preds, targets)

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
        preds = torch.argmax(logits, dim=1).long()
        # logits = logits[:, 1] if self.num_classes == 2 else logits
        self.test_metrics(logits, targets)
        self.log_dict(self.test_metrics, batch_size=batch_size)

        self.cm_test(preds, targets)

        # loss
        self.log("loss/test", loss, batch_size=batch_size)

        batch['loss'] = loss
        batch['repr'] = z.detach().cpu()
        return batch

    def on_test_end(self) -> None:
        fig, ax = self.cm_test.plot()
        self.logger.experiment.log({'test/confusion_matrix': fig})

    def configure_optimizers(self):
        optimizer = optim.Adam([
            {'params': self.head.parameters(), 'lr': self.lr_head},
            {'params': filter(lambda p: p.requires_grad, self.mil.parameters()), 'lr': self.lr_mil}
        ], weight_decay=self.weight_decay)

        return optimizer

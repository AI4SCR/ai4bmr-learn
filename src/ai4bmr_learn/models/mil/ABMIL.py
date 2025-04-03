# copy & paste from: https://github.com/mahmoodlab/TRIDENT/blob/main/trident/slide_encoder_models/load.py and referenced sources

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from einops import rearrange
from torchmetrics import MetricCollection

# %%
from dataclasses import dataclass


class ABMIL(nn.Module):
    """
    Multi-headed attention network with optional gating. Uses tanh-attention and sigmoid-gating as in ABMIL (https://arxiv.org/abs/1802.04712).
    Note that this is different from canonical attention in that the attention scores are computed directly by a linear layer rather than by a dot product between queries and keys.

    Args:
        feature_dim (int): Input feature dimension
        head_dim (int): Hidden layer dimension for each attention head. Defaults to 256.
        n_heads (int): Number of attention heads. Defaults to 8.
        dropout (float): Dropout probability. Defaults to 0.
        n_branches (int): Number of attention branches. Defaults to 1, but can be set to n_classes to generate one set of attention scores for each class.
        gated (bool): If True, sigmoid gating is applied. Otherwise, the simple attention mechanism is used.
    """

    def __init__(self, feature_dim=1024, head_dim=256, n_heads=8, dropout=0.0, n_branches=1, gated=False):
        super().__init__()
        self.gated = gated
        self.n_heads = n_heads

        # Initialize attention head(s)
        self.attention_heads = nn.ModuleList(
            [nn.Sequential(nn.Linear(feature_dim, head_dim), nn.Tanh(), nn.Dropout(dropout)) for _ in range(n_heads)]
        )

        # Initialize gating layers if gating is used
        if self.gated:
            self.gating_layers = nn.ModuleList(
                [
                    nn.Sequential(nn.Linear(feature_dim, head_dim), nn.Sigmoid(), nn.Dropout(dropout))
                    for _ in range(n_heads)
                ]
            )

        # Initialize branching layers
        self.branching_layers = nn.ModuleList([nn.Linear(head_dim, n_branches) for _ in range(n_heads)])

        # Initialize condensing layer if multiple heads are used
        if n_heads > 1:
            self.condensing_layer = nn.Linear(n_heads * feature_dim, feature_dim)

    def forward(self, features, attn_mask=None):
        """
        Forward pass

        Args:
            features (torch.Tensor): Input features, acting as queries and values. Shape: batch_size x num_images x feature_dim
            attn_mask (torch.Tensor): Attention mask to enforce zero attention on empty images. Defaults to None. Shape: batch_size x num_images

        Returns:
            aggregated_features (torch.Tensor): Attention-weighted features aggregated across heads. Shape: batch_size x n_branches x feature_dim
        """

        assert (
            features.dim() == 3
        ), f"Input features must be 3-dimensional (batch_size x num_images x feature_dim). Got {features.shape} instead."
        if attn_mask is not None:
            assert (
                attn_mask.dim() == 2
            ), f"Attention mask must be 2-dimensional (batch_size x num_images). Got {attn_mask.shape} instead."
            assert (
                features.shape[:2] == attn_mask.shape
            ), f"Batch size and number of images must match between features and mask. Got {features.shape[:2]} and {attn_mask.shape} instead."

        # Get attention scores for each head
        head_attentions = []
        head_features = []
        for i in range(len(self.attention_heads)):
            attention_vectors = self.attention_heads[i](
                features
            )  # Main attention vectors (shape: batch_size x num_images x head_dim)

            if self.gated:
                gating_vectors = self.gating_layers[i](
                    features
                )  # Gating vectors (shape: batch_size x num_images x head_dim)
                attention_vectors = attention_vectors.mul(
                    gating_vectors
                )  # Element-wise multiplication to apply gating vectors

            attention_scores = self.branching_layers[i](
                attention_vectors
            )  # Attention scores for each branch (shape: batch_size x num_images x n_branches)

            # Set attention scores for empty images to -inf
            if attn_mask is not None:
                attention_scores = attention_scores.masked_fill(
                    ~attn_mask.unsqueeze(-1), -1e9
                )  # Mask is automatically broadcasted to shape: batch_size x num_images x n_branches

            # Softmax attention scores over num_images
            attention_scores_softmax = F.softmax(attention_scores, dim=1)  # Shape: batch_size x num_images x n_branches

            # Multiply features by attention scores
            weighted_features = torch.einsum(
                "bnr,bnf->brf", attention_scores_softmax, features
            )  # Shape: batch_size x n_branches x feature_dim

            head_attentions.append(attention_scores)
            head_features.append(weighted_features)

        # Concatenate multi-head outputs and condense
        aggregated_features = torch.cat(
            head_features, dim=-1
        )  # Shape: batch_size x n_branches x (n_heads * feature_dim)
        if self.n_heads > 1:
            aggregated_features = self.condensing_layer(
                aggregated_features
            )  # Shape: batch_size x n_branches x feature_dim

        # Stack attention scores
        head_attentions = torch.stack(head_attentions, dim=-1)  # Shape: batch_size x num_images x n_branches x n_heads
        head_attentions = rearrange(
            head_attentions, "b n r h -> b r h n"
        )  # Shape: batch_size x n_branches x n_heads x num_images

        return aggregated_features, head_attentions


# %%
class ABMILClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        feature_dim=1024,
        head_dim=256,
        n_heads=8,
        dropout=0.0,
        n_branches=1,
        gated=False,
        hidden_dim=256,
    ):
        super().__init__()

        self.num_classes = num_classes

        self.pre_attention = nn.Sequential(nn.Linear(feature_dim, feature_dim), nn.GELU(), nn.Dropout(0.1))

        self.abmil = ABMIL(
            n_heads=n_heads,
            feature_dim=feature_dim,
            head_dim=head_dim,
            dropout=dropout,
            n_branches=1,  # no branching
            gated=gated,
        )

        self.post_attention = nn.Sequential(nn.Linear(feature_dim, feature_dim), nn.GELU(), nn.Dropout(0.1))

        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, batch, return_raw_attention=False):
        x = batch["x"]
        x = self.pre_attention(x)
        x, attn = self.abmil(
            x
        )  # Features shape: (b n_branches f), where n_branches = 1. Branching is not used in this implementation.
        x = rearrange(x, "b 1 f -> b f")
        x = self.post_attention(x)  # Attention scores shape: (b r h n), where h is number of attention heads

        logits = self.classifier(x).squeeze(1)

        if return_raw_attention:
            return logits, attn

        return logits


# %%
from torchmetrics.classification import Accuracy, Recall, Precision, F1Score, ConfusionMatrix
from ai4bmr_learn.metrics.classification import get_metric_collection


class ABMILModule(L.LightningModule):

    def __init__(
        self,
        num_classes: int,
        feature_dim=1024,
        head_dim=256,
        n_heads=8,
        dropout=0.0,
        n_branches=1,
        gated=False,
        hidden_dim=256,
        class_weight: torch.tensor = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = ABMILClassifier(
            num_classes=num_classes,
            feature_dim=feature_dim,
            n_heads=n_heads,
            head_dim=head_dim,
            dropout=dropout,
            gated=gated,
            hidden_dim=hidden_dim,
        )

        self.class_weight = class_weight
        self.criterion = nn.CrossEntropyLoss(weight=class_weight)

        # METRICS
        # task = "multiclass" if num_classes > 2 else "binary"
        metrics = get_metric_collection(num_classes=num_classes)
        self.train_metrics = metrics.clone(prefix="train/")
        self.valid_metrics = metrics.clone(prefix="val/")
        self.test_metrics = metrics.clone(prefix="test/")

    def _shared_step(self, batch, batch_idx):
        logits = self.model(batch)
        targets = batch["target"].long()

        loss = self.criterion(logits, targets)

        return logits, targets, loss

    def training_step(self, batch, batch_idx):
        logits, targets, loss = self._shared_step(batch, batch_idx)

        # metrics
        self.train_metrics(logits, targets)
        self.log_dict(self.train_metrics, on_epoch=True, on_step=False)

        # loss
        self.log("train_loss", loss.item(), on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        logits, targets, loss = self._shared_step(batch, batch_idx)

        # metrics
        self.valid_metrics(logits, targets)
        self.log_dict(self.valid_metrics, on_epoch=True, on_step=False)

        # loss
        self.log("val_loss_epoch", loss.item(), on_epoch=True)

    def test_step(self, batch, batch_idx):
        logits, targets, loss = self._shared_step(batch, batch_idx)

        # metrics
        self.test_metrics(logits, targets)
        self.log_dict(self.test_metrics, on_epoch=True, on_step=False)

        # loss
        self.log("test_loss", loss.item())

    def predict_step(self, batch, batch_idx):
        logits = self.model(batch)
        preds = logits.argmax(dim=1)
        batch["prediction"] = preds
        return batch

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=4e-4)
        return optimizer

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class AggregationOutput:
    embedding: torch.Tensor
    weights: torch.Tensor | None = None
    logits: torch.Tensor | None = None


class BaseAggregation(nn.Module):
    output_dim: int

    def __init__(self, input_dim: int):
        super().__init__()
        assert input_dim > 0, "input_dim must be positive"
        self.input_dim = input_dim
        self.output_dim = input_dim

    def validate_inputs(self, bag: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        assert bag.ndim == 3, f"Expected bag [B,N,D], got {tuple(bag.shape)}"
        assert mask.ndim == 2, f"Expected mask [B,N], got {tuple(mask.shape)}"
        assert bag.shape[:2] == mask.shape, "bag and mask shapes differ"
        assert bag.shape[-1] == self.input_dim, f"Expected D={self.input_dim}, got {bag.shape[-1]}"

        mask = mask.bool()
        assert mask.sum(dim=1).min() > 0, "empty bag"
        return mask


class MeanAggregation(BaseAggregation):
    def forward(self, bag: torch.Tensor, mask: torch.Tensor) -> AggregationOutput:
        mask = self.validate_inputs(bag=bag, mask=mask)
        weights = mask.to(dtype=bag.dtype)
        weights = weights / weights.sum(dim=1, keepdim=True)
        embedding = torch.einsum("bn,bnd->bd", weights, bag)
        return AggregationOutput(embedding=embedding, weights=weights)


class MaxAggregation(BaseAggregation):
    def forward(self, bag: torch.Tensor, mask: torch.Tensor) -> AggregationOutput:
        mask = self.validate_inputs(bag=bag, mask=mask)
        masked_bag = bag.masked_fill(~mask.unsqueeze(-1), -torch.inf)
        embedding = masked_bag.amax(dim=1)
        return AggregationOutput(embedding=embedding)


class MinAggregation(BaseAggregation):
    def forward(self, bag: torch.Tensor, mask: torch.Tensor) -> AggregationOutput:
        mask = self.validate_inputs(bag=bag, mask=mask)
        masked_bag = bag.masked_fill(~mask.unsqueeze(-1), torch.inf)
        embedding = masked_bag.amin(dim=1)
        return AggregationOutput(embedding=embedding)


class AttentionAggregation(BaseAggregation):
    """Ilse et al. attention MIL with optional gated attention."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        gated: bool = False,
        projection_dim: int | None = None,
        input_dropout: float = 0.0,
        projection_dropout: float = 0.0,
        attention_dropout: float = 0.0,
    ):
        super().__init__(input_dim=input_dim)
        assert hidden_dim > 0, "hidden_dim must be positive"
        assert projection_dim is None or projection_dim > 0, "projection_dim must be positive"

        self.hidden_dim = hidden_dim
        self.gated = gated

        self.input_dropout = nn.Dropout(input_dropout) if input_dropout > 0 else nn.Identity()

        if projection_dim is None:
            feature_dim = input_dim
            self.projection = nn.Identity()
        else:
            feature_dim = projection_dim
            self.projection = nn.Sequential(
                nn.Linear(input_dim, projection_dim),
                nn.Tanh(),
                nn.Dropout(projection_dropout) if projection_dropout > 0 else nn.Identity(),
            )

        self.output_dim = feature_dim
        self.attention_v = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(attention_dropout) if attention_dropout > 0 else nn.Identity(),
        )
        self.attention_u = (
            nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.Sigmoid(),
                nn.Dropout(attention_dropout) if attention_dropout > 0 else nn.Identity(),
            )
            if gated
            else None
        )
        self.attention_w = nn.Linear(hidden_dim, 1)

    def forward(self, bag: torch.Tensor, mask: torch.Tensor) -> AggregationOutput:
        mask = self.validate_inputs(bag=bag, mask=mask)
        x = self.input_dropout(bag)
        x = self.projection(x)

        attention_features = self.attention_v(x)
        if self.attention_u is not None:
            attention_features = attention_features * self.attention_u(x)

        logits = self.attention_w(attention_features).squeeze(-1)
        logits = logits.masked_fill(~mask, -torch.finfo(logits.dtype).max)
        weights = F.softmax(logits, dim=1)
        embedding = torch.einsum("bn,bnd->bd", weights, x)

        return AggregationOutput(embedding=embedding, weights=weights, logits=logits)

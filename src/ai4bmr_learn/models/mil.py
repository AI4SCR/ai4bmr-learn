from __future__ import annotations

import math
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
    ):
        super().__init__(input_dim=input_dim)
        assert hidden_dim > 0, "hidden_dim must be positive"

        self.hidden_dim = hidden_dim
        self.gated = gated

        self.attention_v = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
        )
        self.attention_u = (
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.Sigmoid(),
            )
            if gated
            else None
        )
        self.attention_w = nn.Linear(hidden_dim, 1)

    def forward(self, bag: torch.Tensor, mask: torch.Tensor) -> AggregationOutput:
        mask = self.validate_inputs(bag=bag, mask=mask)

        attention_features = self.attention_v(bag)
        if self.attention_u is not None:
            attention_features = attention_features * self.attention_u(bag)

        logits = self.attention_w(attention_features).squeeze(-1)
        logits = logits.masked_fill(~mask, -torch.finfo(logits.dtype).max)
        weights = F.softmax(logits, dim=1)
        embedding = torch.einsum("bn,bnd->bd", weights, bag)

        return AggregationOutput(embedding=embedding, weights=weights, logits=logits)


class SimpleAttentionAggregation(BaseAggregation):
    """Single-layer attention MIL aggregation."""

    def __init__(self, input_dim: int):
        super().__init__(input_dim=input_dim)
        self.attention = nn.Linear(input_dim, 1)

    def forward(self, bag: torch.Tensor, mask: torch.Tensor) -> AggregationOutput:
        mask = self.validate_inputs(bag=bag, mask=mask)
        logits = self.attention(bag).squeeze(-1)
        logits = logits.masked_fill(~mask, -torch.finfo(logits.dtype).max)
        weights = F.softmax(logits, dim=1)
        embedding = torch.einsum("bn,bnd->bd", weights, bag)
        return AggregationOutput(embedding=embedding, weights=weights, logits=logits)


class TransformerAttentionAggregation(BaseAggregation):
    """
    One-layer transformer-style MIL aggregation.

    Uses learned query vectors (one per head) to attend over instances.

    Input:
        bag:  [B, N, D]
        mask: [B, N]

    Output:
        embedding: [B, D]
        weights:
            [B, N] if num_heads == 1
            [B, H, N] otherwise
    """

    def __init__(
        self,
        input_dim: int,
        num_heads: int = 1,
        hidden_dim: int | None = None,
        dropout: float = 0.0,
        use_output_projection: bool = True,
    ):
        super().__init__(input_dim=input_dim)

        assert num_heads > 0

        self.num_heads = num_heads
        self.hidden_dim = hidden_dim or (input_dim // num_heads)

        assert self.hidden_dim > 0, "hidden_dim must be positive"

        inner_dim = num_heads * self.hidden_dim

        # Transformer projections
        self.key_proj = nn.Linear(input_dim, inner_dim)
        self.value_proj = nn.Linear(input_dim, inner_dim)

        # Learned global query per head
        self.query = nn.Parameter(torch.randn(num_heads, self.hidden_dim))

        self.dropout = nn.Dropout(dropout)

        if use_output_projection:
            self.output_proj = nn.Linear(inner_dim, input_dim)
        else:
            assert inner_dim == input_dim, "inner_dim must equal input_dim if no output projection"
            self.output_proj = nn.Identity()

        self.output_dim = input_dim

    def forward(
        self,
        bag: torch.Tensor,
        mask: torch.Tensor,
    ) -> AggregationOutput:

        mask = self.validate_inputs(bag=bag, mask=mask)

        B, N, D = bag.shape
        H = self.num_heads
        Hd = self.hidden_dim

        K = self.key_proj(bag).view(B, N, H, Hd)
        V = self.value_proj(bag).view(B, N, H, Hd)

        logits = torch.einsum("hd,bnhd->bhn", self.query, K)
        logits = logits / math.sqrt(Hd)

        logits = logits.masked_fill(
            ~mask[:, None, :],
            -torch.finfo(logits.dtype).max,
        )

        weights = F.softmax(logits, dim=-1)
        weights = self.dropout(weights)

        pooled = torch.einsum("bhn,bnhd->bhd", weights, V)

        # Concatenate heads
        pooled = pooled.reshape(B, H * Hd)

        embedding = self.output_proj(pooled)

        # Match API for single-head case
        if H == 1:
            weights_out = weights.squeeze(1)  # [B, N]
            logits_out = logits.squeeze(1)  # [B, N]
        else:
            weights_out = weights  # [B, H, N]
            logits_out = logits  # [B, H, N]

        return AggregationOutput(
            embedding=embedding,
            weights=weights_out,
            logits=logits_out,
        )

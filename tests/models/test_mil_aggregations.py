import pytest
import torch

from ai4bmr_learn.models.mil import (
    AttentionAggregation,
    MaxAggregation,
    MeanAggregation,
    MinAggregation,
    SimpleAttentionAggregation,
)


def test_masked_pooling_aggregations_ignore_padding():
    bag = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0], [100.0, -100.0]],
            [[5.0, 1.0], [100.0, -100.0], [-100.0, 100.0]],
        ]
    )
    mask = torch.tensor([[True, True, False], [True, False, False]])

    mean = MeanAggregation(input_dim=2)(bag, mask)
    max_ = MaxAggregation(input_dim=2)(bag, mask)
    min_ = MinAggregation(input_dim=2)(bag, mask)

    assert torch.allclose(mean.embedding, torch.tensor([[2.0, 3.0], [5.0, 1.0]]))
    assert torch.allclose(mean.weights, torch.tensor([[0.5, 0.5, 0.0], [1.0, 0.0, 0.0]]))
    assert torch.allclose(max_.embedding, torch.tensor([[3.0, 4.0], [5.0, 1.0]]))
    assert torch.allclose(min_.embedding, torch.tensor([[1.0, 2.0], [5.0, 1.0]]))
    assert max_.weights is None
    assert min_.weights is None


def test_aggregation_rejects_empty_bags():
    bag = torch.randn(2, 3, 4)
    mask = torch.tensor([[True, False, False], [False, False, False]])

    with pytest.raises(AssertionError, match="empty bag"):
        MeanAggregation(input_dim=4)(bag, mask)


@pytest.mark.parametrize("gated", [False, True])
def test_attention_aggregation_masks_and_normalizes_weights(gated):
    bag = torch.randn(2, 4, 6)
    mask = torch.tensor([[True, True, False, False], [True, False, True, False]])

    aggregation = AttentionAggregation(
        input_dim=6,
        hidden_dim=5,
        gated=gated,
    )

    output = aggregation(bag, mask)

    assert output.embedding.shape == (2, 6)
    assert output.weights.shape == (2, 4)
    assert output.logits.shape == (2, 4)
    assert torch.isfinite(output.embedding).all()
    assert torch.isfinite(output.weights).all()
    assert torch.allclose(output.weights.sum(dim=1), torch.ones(2))
    assert torch.all(output.weights[~mask] == 0)


def test_simple_attention_aggregation_masks_and_normalizes_weights():
    bag = torch.randn(2, 4, 6)
    mask = torch.tensor([[True, True, False, False], [True, False, True, False]])

    aggregation = SimpleAttentionAggregation(input_dim=6)

    output = aggregation(bag, mask)

    assert output.embedding.shape == (2, 6)
    assert output.weights.shape == (2, 4)
    assert output.logits.shape == (2, 4)
    assert torch.isfinite(output.embedding).all()
    assert torch.isfinite(output.weights).all()
    assert torch.allclose(output.weights.sum(dim=1), torch.ones(2))
    assert torch.all(output.weights[~mask] == 0)

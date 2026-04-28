import pytest
import torch

from ai4bmr_learn.callbacks.cache import ValidationCache


def test_cache_include_keys_keeps_top_level_keys():
    cache = ValidationCache(save=False, include_keys=["prediction", "ids"])

    cache.accumulate(
        {
            "prediction": torch.ones(2, 3),
            "ids": ["a", "b"],
            "mask": torch.zeros(2, 3, dtype=torch.bool),
        }
    )

    assert len(cache.outputs) == 1
    assert set(cache.outputs[0]) == {"prediction", "ids"}


def test_cache_include_keys_supports_nested_paths():
    cache = ValidationCache(save=False, include_keys=["prediction.logits", "metadata.sample_ids"])

    cache.accumulate(
        {
            "prediction": {"logits": torch.ones(2, 3), "probs": torch.zeros(2, 3)},
            "metadata": {"sample_ids": ["a", "b"], "fold": 0},
            "mask": torch.zeros(2, 3, dtype=torch.bool),
        }
    )

    assert len(cache.outputs) == 1
    assert set(cache.outputs[0]) == {"prediction", "metadata"}
    assert set(cache.outputs[0]["prediction"]) == {"logits"}
    assert set(cache.outputs[0]["metadata"]) == {"sample_ids"}


def test_cache_include_keys_ignores_missing_paths_when_requested():
    cache = ValidationCache(save=False, include_keys=["prediction.logits", "metadata.sample_ids"], ignore_missing=True)

    cache.accumulate(
        {
            "prediction": {"logits": torch.ones(2, 3)},
        }
    )

    assert len(cache.outputs) == 1
    assert set(cache.outputs[0]) == {"prediction"}
    assert set(cache.outputs[0]["prediction"]) == {"logits"}
    assert torch.equal(cache.outputs[0]["prediction"]["logits"], torch.ones(2, 3))


def test_cache_rejects_include_and_exclude_keys_together():
    with pytest.raises(AssertionError, match="mutually exclusive"):
        ValidationCache(save=False, include_keys=["prediction"], exclude_keys=["mask"])


def test_cache_exclude_keys_still_work():
    cache = ValidationCache(save=False, exclude_keys=["prediction.probs", "mask"])

    cache.accumulate(
        {
            "prediction": {"logits": torch.ones(2, 3), "probs": torch.zeros(2, 3)},
            "mask": torch.zeros(2, 3, dtype=torch.bool),
            "ids": ["a", "b"],
        }
    )

    assert len(cache.outputs) == 1
    assert set(cache.outputs[0]) == {"prediction", "ids"}
    assert set(cache.outputs[0]["prediction"]) == {"logits"}

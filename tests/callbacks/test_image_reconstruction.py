from types import SimpleNamespace

import pytest
import torch

from ai4bmr_learn.callbacks.cache import ValidationCache
from ai4bmr_learn.callbacks.image_reconstruction import ImageReconstruction, reconstruction_grid


class DummyLogger:
    def __init__(self) -> None:
        self.calls = []

    def log_image(self, key: str, images: list[torch.Tensor]) -> None:
        self.calls.append({"key": key, "images": images})


def test_reconstruction_grid_accepts_single_channel_mask():
    images = torch.ones(2, 3, 4, 5)
    predictions = torch.zeros(2, 3, 4, 5)
    masks = torch.zeros(2, 1, 4, 5, dtype=torch.bool)
    masks[:, :, 1:3, 2:4] = True

    grid = reconstruction_grid(images, predictions, masks, padding=0)

    assert grid.shape == (3, 8, 15)


def test_reconstruction_grid_accepts_rgb_mask():
    images = torch.ones(2, 3, 4, 5)
    predictions = torch.zeros(2, 3, 4, 5)
    masks = torch.zeros(2, 3, 4, 5, dtype=torch.bool)

    grid = reconstruction_grid(images, predictions, masks, padding=0)

    assert grid.shape == (3, 8, 15)


def test_reconstruction_grid_rejects_non_rgb_images():
    images = torch.ones(2, 1, 4, 5)
    predictions = torch.ones(2, 1, 4, 5)
    masks = torch.zeros(2, 1, 4, 5, dtype=torch.bool)

    with pytest.raises(ValueError, match="Expected RGB images"):
        reconstruction_grid(images, predictions, masks)


def test_reconstruction_grid_rejects_incompatible_mask_shape():
    images = torch.ones(2, 3, 4, 5)
    predictions = torch.ones(2, 3, 4, 5)
    masks = torch.zeros(2, 2, 4, 5, dtype=torch.bool)

    with pytest.raises(ValueError, match="Expected mask shape"):
        reconstruction_grid(images, predictions, masks)


def test_reconstruction_grid_rejects_non_bool_mask():
    images = torch.ones(2, 3, 4, 5)
    predictions = torch.ones(2, 3, 4, 5)
    masks = torch.zeros(2, 1, 4, 5)

    with pytest.raises(ValueError, match="Expected bool mask"):
        reconstruction_grid(images, predictions, masks)


def test_image_reconstruction_collects_num_samples_from_cache():
    cache = ValidationCache(save=False)
    cache.outputs = [
        {
            "image": torch.ones(2, 3, 4, 5),
            "prediction": torch.zeros(2, 3, 4, 5),
            "mask": torch.zeros(2, 1, 4, 5, dtype=torch.bool),
        },
        {
            "image": torch.ones(2, 3, 4, 5),
            "prediction": torch.zeros(2, 3, 4, 5),
            "mask": torch.zeros(2, 1, 4, 5, dtype=torch.bool),
        },
    ]
    callback = ImageReconstruction(num_samples=3)

    images, predictions, masks = callback._collect_tensors(cache)

    assert images.shape == (3, 3, 4, 5)
    assert predictions.shape == (3, 3, 4, 5)
    assert masks.shape == (3, 1, 4, 5)


def test_image_reconstruction_logs_one_grid():
    cache = ValidationCache(save=False)
    cache.outputs = [
        {
            "image": torch.ones(2, 3, 4, 5),
            "prediction": torch.zeros(2, 3, 4, 5),
            "mask": torch.zeros(2, 1, 4, 5, dtype=torch.bool),
        },
    ]
    logger = DummyLogger()
    trainer = SimpleNamespace(callbacks=[cache], logger=logger, sanity_checking=False)
    callback = ImageReconstruction(num_samples=2, padding=0, key="rgb_reconstruction")

    callback.on_validation_epoch_end(trainer, pl_module=None)

    assert len(logger.calls) == 1
    assert logger.calls[0]["key"] == "rgb_reconstruction"
    assert logger.calls[0]["images"][0].shape == (3, 8, 15)

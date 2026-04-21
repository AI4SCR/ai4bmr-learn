import torch
from glom import glom
from lightning.pytorch.callbacks import Callback
from torchvision.utils import make_grid

from ai4bmr_learn.callbacks.cache import ValidationCache


def reconstruction_grid(
    images: torch.Tensor,
    predictions: torch.Tensor,
    masks: torch.Tensor,
    padding: int = 2,
) -> torch.Tensor:
    if images.ndim != 4 or predictions.ndim != 4 or masks.ndim != 4:
        raise ValueError("Expected image/prediction/mask tensors with shape [B, C, H, W].")
    if images.shape != predictions.shape:
        raise ValueError(f"Expected equal image/prediction shapes, got {images.shape}, {predictions.shape}.")
    if images.shape[1] != 3:
        raise ValueError(f"Expected RGB images with shape [B, 3, H, W], got {images.shape}.")
    if masks.dtype != torch.bool:
        raise ValueError(f"Expected bool mask, got {masks.dtype}.")
    if masks.shape[0] != images.shape[0] or masks.shape[-2:] != images.shape[-2:]:
        raise ValueError(f"Expected mask shape [B, 1, H, W] or [B, 3, H, W], got {masks.shape}.")
    if masks.shape[1] == 1:
        masks = masks.expand_as(images)
    elif masks.shape[1] != 3:
        raise ValueError(f"Expected mask shape [B, 1, H, W] or [B, 3, H, W], got {masks.shape}.")

    images = images.detach().float().cpu().clamp(0.0, 1.0)
    predictions = predictions.detach().float().cpu().clamp(0.0, 1.0)
    masks = masks.detach().cpu()

    masked_images = torch.where(masks, torch.zeros_like(images), images)
    reconstructions = torch.where(masks, predictions, images)
    triplets = torch.stack([masked_images, reconstructions, images], dim=1)
    triplets = triplets.flatten(0, 1)
    return make_grid(triplets, nrow=3, normalize=False, padding=padding)


class ImageReconstruction(Callback):
    def __init__(
        self,
        image_key: str = "image",
        prediction_key: str = "prediction",
        mask_key: str = "mask",
        num_samples: int = 5,
        padding: int = 2,
        key: str = "image_reconstruction",
    ) -> None:
        assert num_samples > 0, "num_samples must be positive"
        self.image_key = image_key
        self.prediction_key = prediction_key
        self.mask_key = mask_key
        self.num_samples = num_samples
        self.padding = padding
        self.key = key

    def _get_validation_cache(self, trainer) -> ValidationCache:
        caches = [cb for cb in trainer.callbacks if isinstance(cb, ValidationCache)]
        if len(caches) != 1:
            raise ValueError(f"Expected exactly one ValidationCache callback, found {len(caches)}.")
        return caches[0]

    def _collect_tensors(self, cache: ValidationCache) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not cache.outputs:
            raise ValueError("ValidationCache has no outputs. Add ValidationCache and ensure val_step returns outputs.")

        images = []
        predictions = []
        masks = []
        for output in cache.outputs:
            images.append(glom(output, self.image_key))
            predictions.append(glom(output, self.prediction_key))
            masks.append(glom(output, self.mask_key))
            if sum(image.shape[0] for image in images) >= self.num_samples:
                break

        images = torch.cat(images, dim=0)[: self.num_samples]
        predictions = torch.cat(predictions, dim=0)[: self.num_samples]
        masks = torch.cat(masks, dim=0)[: self.num_samples]

        if images.shape[0] == 0:
            raise ValueError("No samples available in ValidationCache outputs.")
        return images, predictions, masks

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        if trainer.sanity_checking:
            return
        if not hasattr(trainer.logger, "log_image"):
            raise ValueError("Logger does not support log_image.")

        cache = self._get_validation_cache(trainer)
        images, predictions, masks = self._collect_tensors(cache)
        grid = reconstruction_grid(images, predictions, masks, padding=self.padding)
        trainer.logger.log_image(key=self.key, images=[grid])

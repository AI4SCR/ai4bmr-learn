import numpy as np
import torch
from glom import glom
from lightning.pytorch.callbacks import Callback
from torchvision.utils import make_grid

from ai4bmr_learn.callbacks.cache import ValidationCache


class MultiplexedImageReconstruction(Callback):
    def __init__(
        self,
        image_key: str = "image",
        prediction_key: str = "prediction",
        mask_key: str = "mask",
        num_samples: int = 5,
        seed: int = 0,
        padding: int = 2,
    ):
        self.image_key = image_key
        self.prediction_key = prediction_key
        self.mask_key = mask_key
        self.num_samples = num_samples
        self.padding = padding
        self.rng = np.random.default_rng(seed=seed)

    @staticmethod
    def _row_grid(*, masked_ch: torch.Tensor, recon_ch: torch.Tensor, orig_ch: torch.Tensor) -> torch.Tensor:
        if masked_ch.ndim != 2 or recon_ch.ndim != 2 or orig_ch.ndim != 2:
            raise ValueError("Expected channel images with shape [H, W].")
        triplet = torch.stack([masked_ch, recon_ch, orig_ch], dim=0).unsqueeze(1)  # [3, 1, H, W]
        return make_grid(triplet, nrow=3, normalize=True, scale_each=False)

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
            if sum(i.shape[0] for i in images) >= self.num_samples:
                break

        images = torch.cat(images, dim=0)[: self.num_samples]
        predictions = torch.cat(predictions, dim=0)[: self.num_samples]
        masks = torch.cat(masks, dim=0)[: self.num_samples]

        if images.ndim != 4 or predictions.ndim != 4 or masks.ndim != 4:
            raise ValueError("Expected image/prediction/mask tensors with shape [B, C, H, W].")
        if images.shape != predictions.shape or images.shape != masks.shape:
            raise ValueError(
                f"Expected equal shapes for image/prediction/mask, got {images.shape}, {predictions.shape}, {masks.shape}."
            )
        if images.shape[0] == 0:
            raise ValueError("No samples available in ValidationCache outputs.")

        return images, predictions, masks

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        if not hasattr(trainer.logger, "log_image"):
            raise ValueError("Logger does not support log_image.")

        cache = self._get_validation_cache(trainer)
        images, predictions, masks = self._collect_tensors(cache)
        masked_images = images * (1.0 - masks)

        num_samples, num_channels, _, _ = images.shape

        # 1) For each channel: stack num_samples x 3 (masked, recon, orig)
        for c in range(num_channels):
            rows = []
            for s in range(num_samples):
                row = self._row_grid(
                    masked_ch=masked_images[s, c],
                    recon_ch=predictions[s, c],
                    orig_ch=images[s, c],
                )
                rows.append(row)
            grid = make_grid(rows, nrow=1, normalize=False, padding=self.padding)
            trainer.logger.log_image(key=f"mae/multiplexed/by_channel/{c:03d}", images=[grid])

        # 2) For each sample: stack num_channels x 3 (masked, recon, orig)
        for s in range(num_samples):
            rows = []
            for c in range(num_channels):
                row = self._row_grid(
                    masked_ch=masked_images[s, c],
                    recon_ch=predictions[s, c],
                    orig_ch=images[s, c],
                )
                rows.append(row)
            grid = make_grid(rows, nrow=1, normalize=False, padding=self.padding)
            trainer.logger.log_image(key=f"mae/multiplexed/by_sample/{s:03d}", images=[grid])

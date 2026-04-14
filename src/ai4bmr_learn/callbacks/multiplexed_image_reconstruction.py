import torch
from glom import glom
from lightning.pytorch.callbacks import Callback
from PIL import ImageDraw, ImageFont
from torchvision.transforms.functional import to_pil_image, to_tensor
from torchvision.utils import make_grid

from ai4bmr_learn.callbacks.cache import ValidationCache


def normalize_tile(x: torch.Tensor, vmin: float | None = None, vmax: float | None = None) -> torch.Tensor:
    if x.ndim != 2:
        raise ValueError(f"Expected tile with shape [H, W], got {x.shape}.")
    x = x.detach().float().cpu()
    if vmin is None:
        x_min = x.min()
    else:
        x_min = torch.tensor(vmin, dtype=x.dtype)
    if vmax is None:
        x_max = x.max()
    else:
        x_max = torch.tensor(vmax, dtype=x.dtype)
    if (x_max - x_min).item() < 1e-12:
        return torch.zeros_like(x)
    return ((x - x_min) / (x_max - x_min)).clamp(0.0, 1.0)


def row_grid(
    *,
    masked_ch: torch.Tensor,
    recon_ch: torch.Tensor,
    orig_ch: torch.Tensor,
    channel_label: str | None = None,
    label: bool = False,
    scale_each: bool = False,
) -> torch.Tensor:
    if masked_ch.ndim != 2 or recon_ch.ndim != 2 or orig_ch.ndim != 2:
        raise ValueError("Expected channel images with shape [H, W].")
    if scale_each:
        masked = normalize_tile(masked_ch)
        recon = normalize_tile(recon_ch)
        orig = normalize_tile(orig_ch)
    else:
        ref = orig_ch.detach().float().cpu()
        vmin = ref.min().item()
        vmax = ref.max().item()
        masked = normalize_tile(masked_ch, vmin=vmin, vmax=vmax)
        recon = normalize_tile(recon_ch, vmin=vmin, vmax=vmax)
        orig = normalize_tile(orig_ch, vmin=vmin, vmax=vmax)
    if label and channel_label is not None:
        masked = annotate_tile(masked, channel_label)
        recon = annotate_tile(recon, channel_label)
        orig = annotate_tile(orig, channel_label)

    triplet = torch.stack([masked, recon, orig], dim=0).unsqueeze(1)  # [3, 1, H, W]
    return make_grid(triplet, nrow=3, normalize=False, scale_each=False)


def annotate_tile(x: torch.Tensor, text: str) -> torch.Tensor:
    # Draw text after normalization to avoid make_grid normalization distorting labels.
    pil = to_pil_image(x.unsqueeze(0))
    draw = ImageDraw.Draw(pil)
    font = ImageFont.load_default()
    draw.text((2, 2), text, fill=255, font=font)
    return to_tensor(pil).squeeze(0)


class MultiplexedImageReconstruction(Callback):
    def __init__(
        self,
        image_key: str = "image",
        prediction_key: str = "prediction",
        mask_key: str = "mask",
        num_samples: int = 5,
        padding: int = 2,
        channels: list[str] | None = None,
        label: bool = False,
        scale_each: bool = False,
    ):
        self.image_key = image_key
        self.prediction_key = prediction_key
        self.mask_key = mask_key
        self.num_samples = num_samples
        self.padding = padding
        self.channels = channels
        self.label = label
        self.scale_each = scale_each

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

        # NOTE: we set the masked pixels to the minimum value in each channel
        min_per_channel = images.amin(dim=(-2, -1), keepdim=True)
        masked_images = torch.where(masks, min_per_channel, images)
        # plt.imshow(reconstructions[0, 0]).figure.show()

        # DEBUG:
        # reconstructions = torch.where(masks, images, predictions)
        reconstructions = torch.where(masks, predictions, images)
        # plt.imshow(reconstructions[0, 0]).figure.show()

        num_samples, num_channels, _, _ = images.shape
        if self.channels is not None:
            assert len(self.channels) == num_channels, f"Expected {num_channels} channels, got {len(self.channels)}."

        # 1) For each channel: stack num_samples x 3 (masked, recon, orig)
        for c in range(num_channels):
            channel_label = self.channels[c] if self.channels is not None else f"ch_{c:03d}"
            rows = []
            for s in range(num_samples):
                row = row_grid(
                    masked_ch=masked_images[s, c],
                    recon_ch=reconstructions[s, c],
                    orig_ch=images[s, c],
                    channel_label=channel_label,
                    label=self.label,
                    scale_each=self.scale_each,
                )
                rows.append(row)
            grid = make_grid(rows, nrow=1, normalize=False, padding=self.padding)

            if self.channels is not None:
                key = f"by_channel/{self.channels[c]}"
            else:
                key = f"by_channel/{c:03d}"

            trainer.logger.log_image(key=key, images=[grid])

        # 2) For each sample: stack num_channels x 3 (masked, recon, orig)
        for s in range(num_samples):
            rows = []
            for c in range(num_channels):
                channel_label = self.channels[c] if self.channels is not None else f"ch_{c:03d}"
                row = row_grid(
                    masked_ch=masked_images[s, c],
                    recon_ch=reconstructions[s, c],
                    orig_ch=images[s, c],
                    channel_label=channel_label,
                    label=self.label,
                    scale_each=self.scale_each,
                )
                rows.append(row)
            grid = make_grid(rows, nrow=1, normalize=False, padding=self.padding)
            trainer.logger.log_image(key=f"by_sample/{s:03d}", images=[grid])

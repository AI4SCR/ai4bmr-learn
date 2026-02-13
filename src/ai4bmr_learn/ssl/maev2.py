from einops import repeat
import lightning as L

import torch
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

from ai4bmr_learn.models.decoder.masked_decoder import MaskedDecoderDefault, MaskedDecoder
from ai4bmr_learn.ssl.utils import random_token_mask
import torch.nn as nn
from ai4bmr_learn.utils.pooling import pool


class MAEv2(L.LightningModule):
    name = "MAEv2"

    def __init__(
            self,
            *,
            backbone: nn.Module,
            decoder_kwargs: dict | None = None,
            mask_ratio: float = 0.75,
            loss_type: str = "simple",
            channel_weights: list[float] | torch.Tensor | None = None,
            activity_threshold: float = 0,
            weight_loss_by_sparsity: bool = False,
            norm_loss_target: bool = False,
            lr: float = 1.5e-4,
            weight_decay: float = 0.04,
            betas: tuple[float, float] = (0.9, 0.95),
            warmup_steps: int = 2000,
            max_epochs: int = 1500,
            pooling: str | None = "cls",
    ):
        super().__init__()

        self.backbone = backbone
        self.tokenizer = backbone.tokenizer
        self.encoder = backbone.encoder

        decoder_kwargs = decoder_kwargs or {}
        decoder_kwargs = {**MaskedDecoderDefault(num_tokens=self.encoder.num_tokens).model_dump(), **decoder_kwargs}
        self.decoder: MaskedDecoder = MaskedDecoder(**decoder_kwargs)

        if self.encoder.dim != self.decoder.dim:
            self.proj = nn.Linear(self.encoder.dim, self.decoder.dim)
        else:
            self.proj = nn.Identity()

        self.head: nn.Module = nn.Linear(self.decoder.dim, self.tokenizer.num_token_pixels)

        self.mask_ratio = mask_ratio

        # LOSS
        self.loss_type = loss_type
        self.channel_weights = channel_weights
        self.activity_threshold = activity_threshold
        self.weight_loss_by_sparsity = weight_loss_by_sparsity
        self.norm_loss_target = norm_loss_target

        # OPTIMIZER
        self.lr = lr
        self.weight_decay = weight_decay
        self.betas = betas
        self.warmup_steps = warmup_steps
        self.max_epochs = max_epochs
        self.pooling = pooling

        # TRACKING
        self.num_samples_total = 0

        self.save_hyperparameters(ignore=["tokenizer", "encoder", "decoder", "backbone"])

    def _shared_step(self, img):
        batch_size = img.shape[0]

        # 0. tokenize image
        x = self.tokenizer(img)

        # 1. generate random token mask
        # NOTE: the mask is computed for the tokens including prefix tokens (if any)
        num_prefix_tokens = self.encoder.num_prefix_tokens
        num_tokens = self.encoder.num_tokens
        token_mask, _, idx_keep = random_token_mask(
            batch_size=batch_size,
            num_tokens=num_tokens,
            num_prefix_tokens=num_prefix_tokens,
            mask_prefix_tokens=False,
            mask_ratio=self.mask_ratio,
        )

        idx_keep = idx_keep.to(x.device)
        token_mask = token_mask.to(x.device)

        # 2. encode masked patch tokens
        x = self.encoder.forward_masked(x, idx_keep=idx_keep)

        # 3. project to decoder space
        x = self.proj(x)

        # 4. decode masked patch tokens
        x = self.decoder.forward_masked(x, idx_keep=idx_keep)

        # 5. project to pixel space
        x = x[:, 1:]  # remove prefix tokens
        x = self.head(x)
        mask = repeat(token_mask[:, num_prefix_tokens:], "b k -> b k d", d=x.shape[-1])

        # 6. convert tokens to image
        mask = self.tokenizer.tokens2img(mask)
        x = self.tokenizer.tokens2img(x)

        return x, mask

    def _shared_step_unmasked(self, img):
        z = self.backbone(img)
        x = self.proj(z)
        x = self.decoder.forward(x)
        x = x[:, 1:]
        x = self.head(x)
        prediction = self.tokenizer.tokens2img(x)
        mask = torch.ones_like(img)
        return prediction, mask, z

    def _log_stage(self, stage: str, loss: torch.Tensor):
        self.log(f"loss/{stage}", loss, on_step=stage == 'train', on_epoch=True)

        if stage == "train":
            self.log("train/num_samples_total", self.num_samples_total, on_step=True, on_epoch=False)

    def training_step(self, batch, batch_idx):
        images = batch["image"].to(self.device)
        assert not images.isnan().any(), "Input images contain NaNs"

        active_pixels = batch.get("active_pixel_mask", None)
        if active_pixels is not None:
            active_pixels = active_pixels.to(self.device)

        prediction, mask = self._shared_step(images)
        loss = self.compute_loss(img=images, predicted_img=prediction, mask=mask, active_pixels=active_pixels)
        batch_size = images.shape[0]
        self.num_samples_total += batch_size
        self._log_stage(stage="train", loss=loss)
        return loss

    def validation_step(self, batch, batch_idx):
        images = batch["image"].to(self.device)
        assert not images.isnan().any(), "Input images contain NaNs"

        active_pixels = batch.get("active_pixel_mask", None)
        if active_pixels is not None:
            active_pixels = active_pixels.to(self.device)

        prediction, mask, z = self._shared_step_unmasked(images)
        loss = self.compute_loss(img=images, predicted_img=prediction, mask=mask, active_pixels=active_pixels)
        self._log_stage(stage="val", loss=loss)
        return loss

    def test_step(self, batch, batch_idx):
        images = batch["image"].to(self.device)
        assert not images.isnan().any(), "Input images contain NaNs"

        active_pixels = batch.get("active_pixel_mask", None)
        if active_pixels is not None:
            active_pixels = active_pixels.to(self.device)

        prediction_masked, mask_masked = self._shared_step(images)
        loss_masked = self.compute_loss(
            img=images,
            predicted_img=prediction_masked,
            mask=mask_masked,
            active_pixels=active_pixels,
        )

        prediction_unmasked, mask_unmasked, z = self._shared_step_unmasked(images)
        loss_unmasked = self.compute_loss(
            img=images,
            predicted_img=prediction_unmasked,
            mask=mask_unmasked,
            active_pixels=active_pixels,
        )
        self._log_stage(stage="test", loss=loss_unmasked)
        self.log("test/masked_recon_loss", loss_masked, on_step=False, on_epoch=True)

        batch["loss"] = loss_unmasked.item()
        batch["loss_masked"] = loss_masked.item()
        batch["image"] = images.cpu()
        batch["z"] = pool(z, strategy=self.pooling).cpu()
        batch["mask"] = mask_masked.cpu()
        batch["prediction_masked"] = prediction_masked.cpu()
        batch["prediction_unmasked"] = prediction_unmasked.cpu()
        return batch

    def compute_loss_simple(self, *, img, predicted_img, mask):
        assert img.ndim == 4, f"Expected img to have 4 dims [B,C,H,W], got {img.shape}"
        assert predicted_img.shape == img.shape, f"Expected predicted_img shape {img.shape}, got {predicted_img.shape}"
        assert img.shape == mask.shape
        assert mask.mean(dtype=torch.float32) > 0, "Mask must contain at least one active element"

        target = img

        loss = (predicted_img - target) ** 2  # [B, C, H, W]

        # loss on masked pixels scaled by the masking ratio
        loss = loss * mask
        loss = loss.mean(dim=(2, 3))  # per-channel loss [B, C]
        loss = loss.mean() / mask.mean(dtype=torch.float32)

        return loss

    def compute_loss(self, *, img, predicted_img, mask, active_pixels=None):
        match self.loss_type:
            case "simple":
                return self.compute_loss_simple(img=img, predicted_img=predicted_img, mask=mask)
            case "classic":
                return self.compute_classic_loss(
                    img=img,
                    predicted_img=predicted_img,
                    mask=mask,
                    active_pixels=active_pixels,
                )
            case "fg_bg":
                return self.compute_foreground_background_loss(
                    img=img,
                    predicted_img=predicted_img,
                    mask=mask,
                    active_pixels=active_pixels,
                )
            case _:
                raise ValueError(f"Unknown loss_type={self.loss_type}")

    def compute_classic_loss(self, img, predicted_img, mask, active_pixels=None):
        assert img.ndim == 4, f"Expected img to have 4 dims [B,C,H,W], got {img.shape}"
        assert img.shape == mask.shape

        target = img
        batch_size, num_channels, _, _ = target.shape

        if active_pixels is not None:
            channel_activity = active_pixels.mean(dim=(2, 3), dtype=torch.float32)  # [B, C]

        if self.weight_loss_by_sparsity:
            # NOTE: this is computed over the whole image not only the masked pixels, design choice.
            w = 1 / (channel_activity.sqrt() + 1e-3)
            w = w.clamp(min=0.0, max=25.0)  # avoid extreme weights
            w = w / w.mean(dim=1, keepdim=True)
            w = w.to(target.device).float()
        elif self.channel_weights is not None:
            # more than activity_threshold of pixels in each channel must be active to contribute to the loss
            w = self.channel_weights.to(device=target.device, dtype=torch.float32).clone()
            w = (channel_activity > self.activity_threshold) * w.unsqueeze(0)
        else:
            w = torch.ones((batch_size, num_channels), device=target.device, dtype=torch.float32)

        if self.norm_loss_target:
            mean = target.mean(dim=(2, 3), keepdim=True)
            var = target.var(dim=(2, 3), keepdim=True, unbiased=False)

            std = torch.sqrt(var + 1e-6)
            std = std.clamp(min=0.05)
            target = (target - mean) / std

        loss = (predicted_img - target) ** 2  # [B, C, H, W]
        loss = loss * mask

        # sum over H,W to get channel-wise loss
        loss = loss.sum(dim=(2, 3))  # [B, C]
        mask_sum = mask.sum(dim=(2, 3))  # [B, C]
        loss = loss / (mask_sum + 1e-6)

        loss = loss * w
        loss = loss.mean()
        return loss

    def compute_foreground_background_loss(self, img, predicted_img, mask, active_pixels, alpha=0.8):
        assert img.ndim == 4, f"Expected img to have 4 dims [B,C,H,W], got {img.shape}"
        assert img.shape == mask.shape

        mask = mask.bool()
        active_pixels = active_pixels.bool()

        target = img
        batch_size, num_channels, _, _ = target.shape

        if self.channel_weights is not None:
            w = self.channel_weights.to(device=target.device, dtype=torch.float32).clone()
        else:
            w = torch.ones((batch_size, num_channels), device=target.device, dtype=torch.float32)

        per_pixel = (predicted_img - target) ** 2  # [B, C, H, W]

        fg_mask = mask & active_pixels  # [B,C,H,W]
        bg_mask = mask & (~active_pixels)  # [B,C,H,W]

        fg_num = (per_pixel * fg_mask).sum(dim=(2, 3))  # [B,C]
        fg_den = fg_mask.sum(dim=(2, 3)).clamp_min(1.0).to(per_pixel.dtype)  # [B,C]
        fg_mean = fg_num / fg_den  # [B,C]

        bg_num = (per_pixel * bg_mask).sum(dim=(2, 3))  # [B,C]
        bg_den = bg_mask.sum(dim=(2, 3)).clamp_min(1.0).to(per_pixel.dtype)  # [B,C]
        bg_mean = bg_num / bg_den

        loss = alpha * fg_mean + (1 - alpha) * bg_mean
        loss = loss * w
        loss = loss.mean()
        return loss

    def predict_step(self, batch, batch_idx) -> dict:
        images = batch["image"]
        out = self.backbone(images)
        return out

    def configure_optimizers(self):
        if self.warmup_steps < 0:
            raise ValueError(f"warmup_steps must be >= 0, got {self.warmup_steps}")

        total_steps = self.trainer.estimated_stepping_batches
        if total_steps <= self.warmup_steps:
            raise ValueError(
                f"estimated_stepping_batches ({total_steps}) must be > warmup_steps ({self.warmup_steps})"
            )

        decay_params = []
        no_decay_params = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim == 1 or name.endswith(".bias"):
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        optimizer = torch.optim.AdamW(
            [
                {"params": decay_params, "weight_decay": self.weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=self.lr,
            betas=self.betas,
        )

        if self.warmup_steps == 0:
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=total_steps,
                eta_min=0.0,
            )
        else:
            warmup_scheduler = LinearLR(
                optimizer,
                start_factor=0.01,
                end_factor=1.0,
                total_iters=self.warmup_steps,
            )
            cosine_scheduler = CosineAnnealingLR(
                optimizer,
                T_max=total_steps - self.warmup_steps,
                eta_min=0.0,
            )
            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[self.warmup_steps],
            )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

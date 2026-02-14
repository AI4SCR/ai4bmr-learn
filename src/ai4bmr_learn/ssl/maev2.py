from einops import rearrange
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
            fourier_alpha: float = 0.01,
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
        self.fourier_alpha = fourier_alpha

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

        # 5. project to pixel-token space
        x = x[:, 1:]  # remove prefix tokens
        x = self.head(x)

        masked_patch = token_mask[:, num_prefix_tokens:]

        kh, kw = self.tokenizer.kernel_size
        pred_patches = rearrange(
            x,
            "b n (c kh kw) -> b n c kh kw",
            c=self.tokenizer.num_channels,
            kh=kh, kw=kw)
        target_patches = rearrange(img, "b c (h kh) (w kw) -> b (h w) c kh kw", kh=kh, kw=kw)
        return pred_patches, target_patches, masked_patch

    def _shared_step_unmasked(self, img):
        z = self.backbone(img)
        x = self.proj(z)
        x = self.decoder.forward(x)
        x = x[:, 1:]
        x = self.head(x)

        kh, kw = self.tokenizer.kernel_size
        pred_patches = rearrange(
            x,
            "b n (c kh kw) -> b n c kh kw",
            c=self.tokenizer.num_channels,
            kh=kh, kw=kw)
        target_patches = rearrange(img, "b c (h kh) (w kw) -> b (h w) c kh kw", kh=kh, kw=kw)
        masked_patch = torch.ones(
            pred_patches.shape[:2],
            device=pred_patches.device,
            dtype=torch.bool,
        )
        return pred_patches, target_patches, masked_patch, z

    def _log_stage(self, stage: str, loss: torch.Tensor, batch_size: int):
        self.log(f"loss/{stage}", loss, on_step=stage == "train", on_epoch=True, batch_size=batch_size)

        if stage == "train":
            self.log("train/num_samples_total", self.num_samples_total, on_step=True, on_epoch=False, batch_size=batch_size)

    def training_step(self, batch, batch_idx):
        images = batch["image"].to(self.device)
        assert not images.isnan().any(), "Input images contain NaNs"

        prediction_patches, target_patches, masked_patch = self._shared_step(images)
        loss = self.compute_loss(
            target_patches=target_patches,
            predicted_patches=prediction_patches,
            masked_patch=masked_patch,
        )
        batch_size = images.shape[0]
        self.num_samples_total += batch_size
        self._log_stage(stage="train", loss=loss, batch_size=batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        images = batch["image"].to(self.device)
        assert not images.isnan().any(), "Input images contain NaNs"

        prediction_masked_patches, target_patches_masked, masked_patch = self._shared_step(images)
        loss_masked = self.compute_loss(
            target_patches=target_patches_masked,
            predicted_patches=prediction_masked_patches,
            masked_patch=masked_patch,
        )

        prediction_unmasked_patches, target_patches_unmasked, unmasked_patch, _ = self._shared_step_unmasked(images)
        loss_unmasked = self.compute_loss(
            target_patches=target_patches_unmasked,
            predicted_patches=prediction_unmasked_patches,
            masked_patch=unmasked_patch,
        )

        batch_size = images.shape[0]
        self.log("loss/val_masked", loss_masked, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log("loss/val_unmasked", loss_unmasked, on_step=False, on_epoch=True, batch_size=batch_size)
        loss = loss_unmasked
        self._log_stage(stage="val", loss=loss, batch_size=batch_size)
        return loss

    def test_step(self, batch, batch_idx):
        images = batch["image"].to(self.device)
        assert not images.isnan().any(), "Input images contain NaNs"

        prediction_masked_patches, target_patches_masked, masked_patch = self._shared_step(images)
        loss_masked = self.compute_loss(
            target_patches=target_patches_masked,
            predicted_patches=prediction_masked_patches,
            masked_patch=masked_patch,
        )

        prediction_unmasked_patches, target_patches_unmasked, unmasked_patch, z = self._shared_step_unmasked(images)
        loss_unmasked = self.compute_loss(
            target_patches=target_patches_unmasked,
            predicted_patches=prediction_unmasked_patches,
            masked_patch=unmasked_patch,
        )
        batch_size = images.shape[0]
        self._log_stage(stage="test", loss=loss_unmasked, batch_size=batch_size)
        self.log("test/masked_recon_loss", loss_masked, on_step=False, on_epoch=True, batch_size=batch_size)

        h, w = self.tokenizer.grid_size
        prediction_masked = rearrange(prediction_masked_patches, "b (h w) c kh kw -> b c (h kh) (w kw)", h=h, w=w)
        prediction_unmasked = rearrange(prediction_unmasked_patches, "b (h w) c kh kw -> b c (h kh) (w kw)", h=h, w=w)
        mask_patches = masked_patch[:, :, None, None, None].expand_as(target_patches_masked).to(torch.float32)
        mask_img = rearrange(mask_patches, "b (h w) c kh kw -> b c (h kh) (w kw)", h=h, w=w)

        batch["loss"] = loss_unmasked.item()
        batch["loss_masked"] = loss_masked.item()
        batch["image"] = images.cpu()
        batch["z"] = pool(z, strategy=self.pooling).cpu()
        batch["mask"] = mask_img.cpu()
        batch["prediction_masked"] = prediction_masked.cpu()
        batch["prediction_unmasked"] = prediction_unmasked.cpu()
        return batch

    def compute_loss_simple(self, *, target_patches, predicted_patches, masked_patch):
        assert target_patches.ndim == 5, f"Expected target_patches to have 5 dims [B,N,C,Kh,Kw], got {target_patches.shape}"
        assert predicted_patches.shape == target_patches.shape, (
            f"Expected predicted_patches shape {target_patches.shape}, got {predicted_patches.shape}"
        )
        assert masked_patch.shape == target_patches.shape[:2], (
            f"Expected masked_patch shape {target_patches.shape[:2]}, got {masked_patch.shape}"
        )
        assert masked_patch.any().item(), "Mask must contain at least one masked patch"

        sq_error = (predicted_patches - target_patches) ** 2  # [B, N, C, Kh, Kw]
        loss_per_patch_channel = sq_error.mean(dim=(3, 4))  # [B, N, C]

        patch_weights = masked_patch.to(loss_per_patch_channel.dtype).unsqueeze(-1)  # [B, N, 1]
        channel_num = (loss_per_patch_channel * patch_weights).sum(dim=1)  # [B, C]
        channel_den = patch_weights.sum(dim=1)  # [B, 1]
        assert (channel_den > 0).all().item(), "Each sample must contain at least one masked patch"

        loss_per_channel = channel_num / channel_den
        return loss_per_channel.mean()

    def compute_loss_mae_plus(self, *, target_patches, predicted_patches, masked_patch):
        assert target_patches.ndim == 5, f"Expected target_patches to have 5 dims [B,N,C,Kh,Kw], got {target_patches.shape}"
        assert predicted_patches.shape == target_patches.shape, (
            f"Expected predicted_patches shape {target_patches.shape}, got {predicted_patches.shape}"
        )
        assert masked_patch.shape == target_patches.shape[:2], (
            f"Expected masked_patch shape {target_patches.shape[:2]}, got {masked_patch.shape}"
        )
        assert 0.0 <= self.fourier_alpha <= 1.0, f"Expected fourier_alpha in [0,1], got {self.fourier_alpha}"
        assert masked_patch.any().item(), "No masked patches found for MAE+ loss"
        mae_per_patch = ((predicted_patches - target_patches) ** 2).mean(dim=(2, 3, 4))
        loss_mae = mae_per_patch[masked_patch].mean()

        fft_target = torch.fft.fft2(target_patches, dim=(-2, -1))
        fft_pred = torch.fft.fft2(predicted_patches, dim=(-2, -1))
        ft_per_patch = (fft_pred.abs() - fft_target.abs()).abs().mean(dim=(2, 3, 4))
        loss_ft = ft_per_patch[masked_patch].mean()

        return (1 - self.fourier_alpha) * loss_mae + self.fourier_alpha * loss_ft

    def compute_loss(self, *, target_patches, predicted_patches, masked_patch):
        match self.loss_type:
            case "simple":
                return self.compute_loss_simple(
                    target_patches=target_patches,
                    predicted_patches=predicted_patches,
                    masked_patch=masked_patch,
                )
            case "mae_plus":
                return self.compute_loss_mae_plus(
                    target_patches=target_patches,
                    predicted_patches=predicted_patches,
                    masked_patch=masked_patch,
                )
            case "classic":
                h, w = self.tokenizer.grid_size
                target_img = rearrange(target_patches, "b (h w) c kh kw -> b c (h kh) (w kw)", h=h, w=w)
                predicted_img = rearrange(predicted_patches, "b (h w) c kh kw -> b c (h kh) (w kw)", h=h, w=w)
                mask_patches = masked_patch[:, :, None, None, None].expand_as(target_patches).to(torch.float32)
                mask_img = rearrange(mask_patches, "b (h w) c kh kw -> b c (h kh) (w kw)", h=h, w=w)
                return self.compute_classic_loss(
                    img=target_img,
                    predicted_img=predicted_img,
                    mask=mask_img,
                )
            case _:
                raise ValueError(f"Unknown loss_type={self.loss_type}")

    def compute_classic_loss(self, img, predicted_img, mask):
        assert img.ndim == 4, f"Expected img to have 4 dims [B,C,H,W], got {img.shape}"
        assert img.shape == mask.shape

        target = img
        batch_size, num_channels, _, _ = target.shape

        if self.weight_loss_by_sparsity:
            raise ValueError("weight_loss_by_sparsity is disabled in MAEv2")
        elif self.channel_weights is not None:
            w = self.channel_weights.to(device=target.device, dtype=torch.float32).clone()
            w = w.unsqueeze(0).expand(batch_size, -1)
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
                start_factor=0.1,
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

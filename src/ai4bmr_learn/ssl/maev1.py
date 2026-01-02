from einops import repeat
import lightning as L

import torch
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

from ai4bmr_learn.models.decoder.masked_decoder import MaskedDecoderDefault, MaskedDecoder
from ai4bmr_learn.ssl.utils import random_token_mask
import torch.nn as nn
from ai4bmr_learn.utils.pooling import pool

class MAEv1(L.LightningModule):
    name = "MAEv1"

    def __init__(
            self,
            *,
            backbone: nn.Module,
            decoder_kwargs: dict | None = None,
            mask_ratio: float = 0.75,
            weight_loss_by_sparsity: bool = False,
            norm_loss_target: bool = False,
            lr: float = 1.5e-4,
            weight_decay: float = 0.04,
            betas: tuple[float, float] = (0.9, 0.95),
            warmup_lr_epochs: int = 200,
            max_epochs: int = 1500,
            pooling: str | None = 'cls',
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
        self.weight_loss_by_sparsity = weight_loss_by_sparsity
        self.norm_loss_target = norm_loss_target

        # OPTIMIZER
        self.lr = lr
        self.weight_decay = weight_decay
        self.betas = betas
        self.warmup_lr_epochs = warmup_lr_epochs
        self.max_epochs = max_epochs
        self.pooling = pooling

        # TRACKING
        self.losses = []
        self.num_samples_epoch = 0
        self.num_samples_total = 0
        self.batch = 0

        self.save_hyperparameters(ignore=['tokenizer', 'encoder', 'decoder', 'backbone'])

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

        # 3. decode masked patch tokens
        x = self.decoder.forward_masked(x, idx_keep=idx_keep)

        # 4. project to pixel space
        x = x[:, 1:]  # remove prefix tokens
        x = self.head(x)
        mask = repeat(token_mask[:, num_prefix_tokens:], "b k -> b k d", d=x.shape[-1])

        # 5. convert tokens to image
        mask = self.tokenizer.tokens2img(mask)
        x = self.tokenizer.tokens2img(x)

        return x, mask

    def training_step(self, batch, batch_idx):
        images = batch["image"].to(self.device)
        predicted_img, mask = self._shared_step(images)
        loss = self.compute_loss(img=images, predicted_img=predicted_img, mask=mask)

        batch_size = num_samples_batch = images.shape[0]
        self.num_samples_epoch += num_samples_batch
        self.num_samples_total += num_samples_batch
        self.losses.append(loss.item())

        self.logger.experiment.log(
            {
                "loss/train_mae_batch": loss.item(),
                "train/num_samples_batch": num_samples_batch,
                "train/num_samples_total": self.num_samples_total,
                "train/batch": self.batch,
                "train/batch_idx": batch_idx,
                "trainer/global_step": self.trainer.global_step,
            }
        )

        self.log("loss/train", loss, on_step=True, on_epoch=True, batch_size=batch_size,
        )
        self.batch += 1
        return loss

    def on_train_epoch_start(self) -> None:
        self.losses = []
        self.num_samples_epoch = 0

    def on_train_epoch_end(self) -> None:
        avg_loss = sum(self.losses) / len(self.losses)
        self.logger.experiment.log(
            {
                "loss/train_mae_epoch": avg_loss,
                "train/num_samples_epoch": self.num_samples_epoch,
                "trainer/global_step": self.trainer.global_step,
                "epoch": self.trainer.current_epoch,
            }
        )

    def validation_step(self, batch, batch_idx):
        images = batch["image"]

        # UNMASKED PREDICTIONS
        z = self.backbone(images)
        x = self.proj(z)
        x = self.decoder.forward(x)
        x = x[:, 1:]
        x = self.head(x)
        predictions = self.tokenizer.tokens2img(x)
        mask = torch.ones_like(images)  # all pixels contribute to loss
        loss = self.compute_loss(img=images, predicted_img=predictions, mask=mask)

        # MASKED PREDICTIONS
        predictions_masked, masks = self._shared_step(images)
        loss_masked = self.compute_loss(img=images, predicted_img=predictions_masked, mask=masks)

        batch_size = num_samples_batch = images.shape[0]

        self.logger.experiment.log(
            {
                "loss/val_mae_batch": loss.item(),
                "loss/val_mae_masked_batch": loss_masked.item(),
                "val/num_samples_batch": num_samples_batch,
                "trainer/global_step": self.trainer.global_step,
            }
        )
        self.log("loss/val", loss, batch_size=batch_size)

        batch['loss'] = loss.item()
        batch['loss_masked'] = loss_masked.item()
        batch["image"] = images.detach().cpu()
        batch['embedding'] = pool(z, strategy=self.pooling).detach().cpu()
        batch['mae'] = {'prediction': predictions.detach().cpu(),
                        'prediction_masked': predictions_masked.detach().cpu(),
                        'masks': masks.detach().cpu()}
        return batch

    def compute_loss_legacy(self, img, predicted_img, mask):
        # TODO: add loss with channel weights
        # TODO: add per-patch normalization
        loss = torch.mean((predicted_img - img) ** 2 * mask) / self.mask_ratio  # L2
        # loss = torch.mean((predicted_img - img).abs() * mask) / self.mask_ratio  # L1
        return loss

    def compute_loss(self, img, predicted_img, mask):
        assert img.ndim == 4, f"Expected img to have 4 dims [B,C,H,W], got {img.shape}"
        assert img.shape == mask.shape

        target = img
        B, C, H, W = target.shape

        if self.weight_loss_by_sparsity:
            # NOTE: this is computed over the whole image not only the masked pixels, design choice.
            pc = (target > 0).mean(dim=(2, 3), dtype=torch.float32)

            wc = 1 / ( pc.sqrt() + 1e-3 )
            wc = wc.clamp(min=0.0, max=15.0)  # avoid extreme weights
            wc = wc.to(target.device).float()
            wc = wc / wc.mean(dim=1, keepdim=True)

        else:
            wc = torch.ones((B, C), device=target.device, dtype=torch.float32)

        if self.norm_loss_target:
            mean = target.mean(dim=(2, 3), keepdim=True)
            var = target.var(dim=(2, 3), keepdim=True, unbiased=False)

            std = torch.sqrt(var + 1e-6)
            std = std.clamp(min=0.05)

            target = (target - mean) / std

        loss = (predicted_img - target) ** 2  # [B, C, H, W]

        # loss on masked pixels scaled by the masking ratio
        loss = (loss * mask)

        loss = loss.mean(dim=(2, 3))  # per-channel loss [B, C]
        loss = loss * wc

        loss = loss.mean() / mask.mean(dtype=torch.float32)

        return loss


    def predict_step(self, batch, batch_idx) -> dict:
        images = batch['image']
        out = self.backbone(images)
        return out

    # https://github.com/facebookresearch/mae/blob/main/PRETRAIN.md
    def configure_optimizers(self):

        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            betas=self.betas,
            weight_decay=self.weight_decay,
            # https://towardsdatascience.com/weight-decay-and-its-peculiar-effects-66e0aee3e7b8/
            # eps=1e-3
        )

        # 1) Warm up from lr*1e-6 → lr over warmup_lr_epochs
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=self.warmup_lr_epochs,
        )
        # 2) Cosine decay from lr → 0 over (max_epochs - warmup_lr_epochs) epochs
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.max_epochs - self.warmup_lr_epochs,
            eta_min=0.0,
        )
        # Chain them together at the warmup boundary
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[self.warmup_lr_epochs],
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def on_train_end(self) -> None:

        if self.trainer.fast_dev_run:
            return

        # TODO: should be a callback
        model_ckpt_cb = list(filter(lambda x: isinstance(x, L.pytorch.callbacks.ModelCheckpoint), self.trainer.callbacks))
        if len(model_ckpt_cb) > 0:
            cb = model_ckpt_cb[0]
            best_model_path = cb.best_model_path
            last_model_path = cb.last_model_path
            self.trainer.logger.experiment.config.update({
                'best_model_path': best_model_path,
                'last_model_path': last_model_path,
            })

        # TODO: should be a callback
        run_id = self.trainer.logger.experiment.id
        run_name = self.trainer.logger.experiment.name
        self.trainer.logger.experiment.config.update({'run_id': run_id, 'run_name': run_name})


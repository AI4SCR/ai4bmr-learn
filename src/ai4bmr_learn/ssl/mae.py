from einops import repeat
import lightning as L
import torch
import torch.nn as nn

from ai4bmr_learn.models.decoder.masked_decoder import MaskedDecoderDefault, MaskedDecoder
from ai4bmr_learn.ssl.utils import random_token_mask

from ai4bmr_learn.models.backbones.base_backbone import BaseBackbone


class MAE(L.LightningModule):
    name = "MAE"

    def __init__(
            self,
            *,
            backbone: BaseBackbone,
            decoder_kwargs: dict | None = None,
            mask_ratio: float = 0.75,
            batch_size: int = 512,
            accumulate_grad_batches: int = 8,
            base_learning_rate=1.5e-4,
            weight_decay: float = 0.05,
            warmup_epochs: int = 200,
            epochs: int = 1000,
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

        self.batch_size = batch_size
        self.accumulate_grad_batches = accumulate_grad_batches
        self.effective_batch_size = batch_size * accumulate_grad_batches
        self.base_learning_rate = base_learning_rate
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs
        self.epochs = epochs
        self.mask_ratio = mask_ratio

        self.losses = []
        self.num_samples_epoch = 0
        self.num_samples_total = 0
        self.batch = 0

        self.save_hyperparameters(ignore=['backbone'])

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
                "train/mae_loss_batch": loss.item(),
                "train/num_samples_batch": num_samples_batch,
                "train/num_samples_total": self.num_samples_total,
                "train/batch": self.batch,
                "train/batch_idx": batch_idx,
                "trainer/global_step": self.trainer.global_step,
            }
        )

        self.log(
            "train_loss_epoch",
            loss,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
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
                "train/mae_loss_epoch": avg_loss,
                "train/num_samples_epoch": self.num_samples_epoch,
                "trainer/global_step": self.trainer.global_step,
                "epoch": self.trainer.current_epoch,
            }
        )

    def validation_step(self, batch, batch_idx):
        images = batch["image"]
        predictions, masks = self._shared_step(images)
        loss = self.compute_loss(images, predictions, masks)

        batch_size = num_samples_batch = images.shape[0]

        self.logger.experiment.log(
            {
                "val/mae_loss_batch": loss.item(),
                "val/num_samples_batch": num_samples_batch,
                "trainer/global_step": self.trainer.global_step,
            }
        )
        self.log(
            "val_loss_epoch", loss, on_step=False, on_epoch=True, batch_size=batch_size
        )

    def compute_loss(self, img, predicted_img, mask):
        loss = torch.mean((predicted_img - img) ** 2 * mask) / self.mask_ratio  # L2
        # loss = torch.mean((predicted_img - img).abs() * mask) / self.mask_ratio  # L1
        return loss

    def configure_optimizers(self):
        import math

        # https://github.com/facebookresearch/mae/blob/main/PRETRAIN.md
        optim = torch.optim.AdamW(
            self.parameters(),
            lr=self.base_learning_rate * self.effective_batch_size / 256,
            betas=(0.9, 0.95),
            weight_decay=self.weight_decay,
        )
        lr_func = lambda epoch: min(
            (epoch + 1) / (self.warmup_epochs + 1e-8),
            0.5 * (math.cos(epoch / self.epochs * math.pi) + 1),
        )
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_func)
        return {
            "optimizer": optim,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }


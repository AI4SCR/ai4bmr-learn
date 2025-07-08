from einops import repeat
import lightning as L
import torch
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

from ai4bmr_learn.models.decoder.masked_decoder import MaskedDecoderDefault, MaskedDecoder
from ai4bmr_learn.ssl.utils import random_token_mask
import torch.nn as nn


class MAEv1(L.LightningModule):
    name = "MAEv1"

    def __init__(
            self,
            *,
            backbone: nn.Module,
            decoder_kwargs: dict | None = None,
            mask_ratio: float = 0.75,
            lr: float = 1.5e-4,
            global_batch_size: int = 512 * 8,
            weight_decay: float = 0.04,
            betas: tuple[float, float] = (0.9, 0.95),
            warmup_lr_epochs: int = 200,
            max_epochs: int = 150,
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

        # OPTIMIZER
        self.lr = lr
        self.global_batch_size = global_batch_size
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

        self.save_hyperparameters(ignore=['backbone'])

    def pool(self, x):
        if self.pooling is None:
            return x
        elif self.pooling == 'cls':
            return x[:, 0]
        elif self.pooling == 'flatten':
            return x.flatten(start_dim=1)
        else:
            raise NotImplementedError(f'{self.pooling} is not implemented.')

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

        # UNMASKED PREDICTIONS
        z = self.backbone(images)
        x = self.proj(z)
        x = self.decoder.forward(x)
        x = x[:, 1:]
        x = self.head(x)
        predictions = self.tokenizer.tokens2img(x)
        loss = self.compute_loss(img=images, predicted_img=predictions, mask=1)

        # MASKED PREDICTIONS
        predictions_masked, masks = self._shared_step(images)
        loss_masked = self.compute_loss(img=images, predicted_img=predictions_masked, mask=masks)

        batch_size = num_samples_batch = images.shape[0]

        self.logger.experiment.log(
            {
                "val/mae_loss_batch": loss.item(),
                "val/mae_loss_masked_batch": loss_masked.item(),
                "val/num_samples_batch": num_samples_batch,
                "trainer/global_step": self.trainer.global_step,
            }
        )
        self.log("val_loss_epoch", loss, on_step=False, on_epoch=True, batch_size=batch_size)

        batch['loss'] = loss.item()
        batch['loss_masked'] = loss_masked.item()
        batch["image"] = images.detach().cpu()
        batch['embedding'] = self.pool(z).detach().cpu()
        batch['mae'] = {'prediction': predictions.detach().cpu(),
                        'prediction_masked': predictions_masked.detach().cpu(),
                        'masks': masks.detach().cpu()}
        return batch

    def compute_loss(self, img, predicted_img, mask):
        loss = torch.mean((predicted_img - img) ** 2 * mask) / self.mask_ratio  # L2
        # loss = torch.mean((predicted_img - img).abs() * mask) / self.mask_ratio  # L1
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

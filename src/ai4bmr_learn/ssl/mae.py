from einops import repeat
import lightning as L
import torch
import torch.nn as nn

from ai4bmr_learn.models.encoder.masked_encoder import MaskedEncoderDefault, MaskedEncoder
from ai4bmr_learn.models.decoder.masked_decoder import MaskedDecoderDefault, MaskedDecoder
from ai4bmr_learn.models.tokenizer.conv2d import TokenizerConv, TokenizerConvConfig
from ai4bmr_learn.ssl.utils import random_token_mask

class Backbone(nn.Module):

    def __init__(self, image_size: int = 224):
        super().__init__()
        self.image_size = image_size

        # tokenizer
        kwargs = TokenizerConvConfig(image_size=image_size).model_dump()
        self.tokenizer = TokenizerConv(**kwargs)

        # encoder
        kwargs = MaskedEncoderDefault(num_patches=self.tokenizer.num_tokens).model_dump()
        self.encoder = MaskedEncoder(**kwargs)

    def forward(self, x):
        x = self.tokenizer(x)
        x = self.encoder(x)
        return x

    def forward_masked(self, x, idx_keep):
        x = self.tokenizer(x)
        x = self.encoder.forward_masked(x, idx_keep=idx_keep)
        return x


class MAE(L.LightningModule):
    name = "MAE"

    def __init__(
        self,
        *,
        image_size: int = 224,
        mask_ratio: float = 0.75,
        batch_size: int = 512,
        accumulate_grad_batches: int = 8,
        base_learning_rate=1.5e-4,
        weight_decay: float = 0.05,
        warmup_epochs: int = 200,
        epochs: int = 2000,
    ):
        super().__init__()


        self.backbone: MaskedEncoder = Backbone(image_size=image_size)

        kwargs = MaskedDecoderDefault(num_tokens=self.backbone.num_tokens).model_dump()
        self.decoder: MaskedDecoder = MaskedDecoder(**kwargs)

        if self.backbone.dim != self.decoder.dim:
            self.proj = nn.Linear(self.backbone.dim, self.decoder.dim)
        else:
            self.proj = nn.Identity()

        self.head: nn.Module = nn.Linear(self.decoder.dim, self.tokenizer.num_patch_pixels)

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

        self.save_hyperparameters()

    def register_model(self, model):
        self.tokenizer = model.tokenizer
        self.encoder = model.encoder
        self.proj = model.proj
        self.decoder = model.decoder
        self.head = model.head

    def register_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    def register_encoder(self, encoder):
        self.encoder = encoder

    def register_proj(self, proj):
        self.proj = proj

    def register_decoder(self, decoder):
        self.decoder = decoder

    def register_head(self, head):
        self.head = head

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

        if self.current_epoch % 50 == 0 and batch_idx == 0:
            print(
                f"Logging samples at Epoch {self.current_epoch} and batch {batch_idx}"
            )
            num_samples = 4
            num_channels = images.shape[1]

            images = images[:num_samples].detach().cpu().float()
            predictions = predictions[:num_samples].detach().cpu().float()
            masks = masks[:num_samples].detach().cpu()

            channel_names = batch.get("channel_names", None)
            if channel_names is not None:
                channel_names = [i[0] for i in channel_names]
            else:
                channel_names = [f"channel_{i}" for i in range(num_channels)]

            self.log_composition(
                images=images,
                predictions=predictions,
                masks=masks,
                channel_names=channel_names,
            )
            if num_channels == 3:
                self.log_composition_3D(
                    images=images,
                    predictions=predictions,
                    masks=masks,
                    channel_names=channel_names,
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
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optim, lr_lambda=lr_func, verbose=True
        )
        return {
            "optimizer": optim,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def log_reconstructions(
        self, *, images, predictions, masks, channel_names: list[str] = None
    ):
        from torchvision.utils import make_grid

        # full reconstruction
        reconstruction = predictions.clone()
        reconstruction[~masks.bool()] = images[~masks.bool()]

        grid = make_grid(reconstruction, nrow=3)
        self.logger.log_image(
            key=f"reconstruction-v1/",
            images=[grid],
            caption=["Reconstruction"],
        )

    def log_composition(self, images, predictions, channel_names, masks=None):

        for i, (img, pred, mask) in enumerate(zip(images, predictions, masks)):
            img_cols, caption = grid_composition(
                img, pred, mask, channel_names=channel_names
            )

            C, height, width = img.shape
            img = img[mask.bool()].reshape(C, -1)
            pred = pred[mask.bool()].reshape(C, -1)

            hist_col = image_to_dist_grid(
                image=img,
                prediction=pred,
                channel_names=channel_names,
                kind="hist",
                height=height,
                width=width,
                nrow=1,
                return_list=False,
            )

            grid = torch.cat([img_cols, hist_col], dim=-1)
            caption += " | Histogram"

            self.logger.log_image(
                key=f"composition-v1/{i}",
                images=[grid],
                caption=[caption],
            )

    def log_multi_channel_grid(self, images, predictions, channel_names, masks=None):

        for i, (img, pred, mask) in enumerate(zip(images, predictions, masks)):
            imgs, caps = grid_composition(img, pred, mask, channel_names=channel_names)

            self.logger.log_image(
                key=f"reconstructions-v1/{i}",
                images=[imgs],
                caption=[caps],
            )

    def log_channel_histograms(
        self, *, images, predictions, masks, channel_names: list[str]
    ):
        from visualization.multi_channel_image import image_to_dist_grid

        for i, (img, pred, mask) in enumerate(zip(images, predictions, masks)):
            # NOTE: we only keep the masked pixels
            C, height, width = img.shape
            img = img[mask.bool()].reshape(C, -1)
            pred = pred[mask.bool()].reshape(C, -1)

            hist_list = image_to_dist_grid(
                image=img,
                prediction=pred,
                channel_names=channel_names,
                kind="hist",
                height=height,
                width=width,
                return_list=True,
            )

            self.logger.log_image(
                key=f"histograms/",
                images=hist_list,
                caption=channel_names,
            )


from visualization.multi_channel_image import image_to_channels_grid, image_to_dist_grid


def grid_composition(img, pred, mask, channel_names: list[str] = None):
    g0, _, _ = image_to_channels_grid(
        img,
        nrow=1,
        normalize=True,
        scale_each=True,
        cmap_name="inferno",
        channel_names=channel_names,
    )

    if mask is not None:
        pred[~mask.bool()] = 0
        g1, _, _ = image_to_channels_grid(
            pred,
            nrow=1,
            normalize=True,
            scale_each=True,
            cmap_name="inferno",
            channel_names=channel_names,
        )

        img_masked = img.clone()
        img_masked[mask.bool()] = 0

        g2, _, _ = image_to_channels_grid(
            img_masked,
            nrow=1,
            normalize=True,
            scale_each=True,
            cmap_name="inferno",
            channel_names=channel_names,
        )

        reconstruction = pred.clone()
        reconstruction[~mask.bool()] = img[~mask.bool()]

        g3, _, _ = image_to_channels_grid(
            reconstruction,
            nrow=1,
            normalize=True,
            scale_each=True,
            cmap_name="inferno",
            channel_names=channel_names,
        )

        g = torch.cat([g0, g3, g1, g2], dim=2)
        caption = "Image | Reconstruction | Prediction | Masked Prediction"
    else:
        g1, _, _ = image_to_channels_grid(
            pred,
            nrow=1,
            normalize=True,
            scale_each=True,
            cmap_name="inferno",
            channel_names=channel_names,
        )

        g = torch.cat([g0, g1], dim=2)
        caption = "Image | Reconstruction"

    return g, caption

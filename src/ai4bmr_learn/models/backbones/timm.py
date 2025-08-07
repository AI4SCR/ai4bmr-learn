import timm
import torch
import torch.nn as nn
from pathlib import Path
from loguru import logger

# %% BACKBONE
class Backbone(nn.Module):
    """A wrapper around timm.create_model to create a backbone model."""

    def __init__(self,
                 model_name: str = 'vit_small_patch16_224',
                 num_channels: int = 3,
                 num_classes: int = 0,
                 pretrained: bool = False,
                 global_pool: str = "token",
                 # ViT
                 # img_size: int = 224,
                 # dynamic_img_size: bool = True,
                 ckpt_path: Path | None = None,
                 **kwargs
                 ):
        """Initializes the Backbone model.

        Args:
            model_name: The name of the timm model to create.
            num_channels: The number of input channels.
            num_classes: The number of output classes. If 0, the classifier is removed.
            pretrained: Whether to load pretrained weights from timm.
            global_pool: The type of global pooling to use in the model.
            ckpt_path: Optional path to a checkpoint file to load.
            **kwargs: Additional arguments to pass to timm.create_model.
        """
        super().__init__()

        self.backbone = timm.create_model(model_name=model_name,
                                          num_classes=num_classes,
                                          global_pool=global_pool,
                                          in_chans=num_channels,
                                          pretrained=pretrained,
                                          **kwargs)

        if ckpt_path is not None:
            state_dict = torch.load(ckpt_path, weights_only=True, map_location=torch.device('cpu'))
            missing, unexpected = self.backbone.load_state_dict(state_dict=state_dict, strict=True)
            if missing:
                logger.warning(f'checkpoint has missing keys: {missing}')
            if unexpected:
                logger.warning(f'checkpoint has unexpected keys: {unexpected}')

    def forward(self, x):
        x = self.backbone(x)
        return x


# %% MaskedAutoEncoder
from ai4bmr_learn.models.tokenizer.base import BaseTokenizer
from ai4bmr_learn.models.encoder.masked_encoder import BaseMaskedEncoder
from ai4bmr_learn.models.utils import get_at_index


class Tokenizer(BaseTokenizer):
    """Image tokenizer using the patch embedding layer of a Vision Transformer."""
    def __init__(self, model, image_size: int):
        """Initializes the Tokenizer.

        Args:
            model: A timm Vision Transformer model.
            image_size: The size of the input image (assumed to be square).
        """
        kernel_size = model.patch_embed.proj.kernel_size
        dim = model.patch_embed.proj.out_channels
        num_channels = model.patch_embed.proj.in_channels

        super().__init__(
            image_size=image_size,
            kernel_size=kernel_size,
            dim=dim,
            num_channels=num_channels,
        )

        self.model = model.patch_embed

        assert self.model.proj.kernel_size == self.kernel_size
        assert self.model.proj.stride == self.stride

    def forward(self, x):
        return self.model(x)

    def tokens2img(self, x: torch.Tensor) -> torch.Tensor:
        from einops import rearrange

        kh, kw = self.kernel_size
        h, w = self.grid_size
        img = rearrange(x, "b (h w) (c p1 p2) -> b c (h p1) (w p2)", p1=kh, p2=kw, h=h, w=w)
        return img


class MaskedEncoder(BaseMaskedEncoder):
    """A masked encoder for a Vision Transformer.

    This encoder processes only a subset of the input tokens, as required for
    masked autoencoding. It removes the classification head from the model.
    """
    def __init__(self, model, num_patches: int):
        """Initializes the MaskedEncoder.

        Args:
            model: A timm Vision Transformer model.
            num_patches: The total number of patches in the image.
        """
        self.num_patches = num_patches
        self.num_prefix_tokens = model.num_prefix_tokens
        self.dim = model.embed_dim

        super().__init__(num_tokens=self.num_patches + self.num_prefix_tokens)

        self.model = model

        # remove unused layers
        # del self.model.patch_embed  # needed for dynamic_image_size = True
        del self.model.fc_norm
        del self.model.head_drop
        del self.model.head

    def forward_encoder(self, x):
        x = self.model.patch_drop(x)
        x = self.model.norm_pre(x)
        x = self.model.blocks(x)
        x = self.model.norm(x)
        return x

    def forward(self, x):
        x = self.model._pos_embed(x)
        x = self.forward_encoder(x)
        return x

    def forward_masked(self, x, idx_keep: torch.Tensor):
        x = self.model._pos_embed(x)
        x = get_at_index(x, index=idx_keep.to(x.device))
        x = self.forward_encoder(x)
        return x


class MaskedAutoEncoder(nn.Module):
    """A Masked Autoencoder (MAE) model built from a timm Vision Transformer."""

    def __init__(self,
                 model_name: str = 'vit_small_patch16_224',
                 num_channels: int = 3,
                 num_classes: int = 0,
                 pretrained: bool = False,
                 global_pool: str = "token",
                 # ViT
                 # img_size: int = 224,
                 # dynamic_img_size: bool = True,
                 **kwargs
                 ):
        """Initializes the MaskedAutoEncoder model.

        This constructor creates a Masked Autoencoder (MAE) model from a
        timm Vision Transformer. It instantiates a tokenizer and a masked
        encoder.

        Args:
            model_name: The name of the timm model to create.
            num_channels: The number of input channels.
            num_classes: The number of output classes.
            pretrained: Whether to load pretrained weights.
            global_pool: The type of global pooling to use.
            **kwargs: Additional arguments to pass to timm.create_model.
                      Note that `img_size` is an important argument to pass here.
        """
        super().__init__()

        self.backbone = timm.create_model(model_name=model_name,
                                          num_classes=num_classes,
                                          global_pool=global_pool,
                                          in_chans=num_channels,
                                          pretrained=pretrained,
                                          **kwargs)

        self.tokenizer = Tokenizer(self.backbone, image_size=kwargs.get('img_size'))
        self.encoder = MaskedEncoder(self.backbone, num_patches=self.tokenizer.num_tokens)

    def forward(self, x):
        x = self.tokenizer(x)
        x = self.encoder(x)
        return x

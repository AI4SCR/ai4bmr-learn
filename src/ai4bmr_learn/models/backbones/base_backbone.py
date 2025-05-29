import torch.nn as nn
from ai4bmr_learn.models.tokenizer.base import BaseTokenizer
from ai4bmr_learn.models.encoder.base import BaseMaskedEncoder


class BaseBackbone(nn.Module):
    """Generic Backbone that combines a tokenizer and encoder."""

    def __init__(self, tokenizer: BaseTokenizer, encoder: BaseMaskedEncoder):
        super().__init__()
        self.tokenizer = tokenizer
        self.encoder = encoder

    def forward(self, x):
        x = self.tokenizer(x)
        x = self.encoder(x)
        return x

    def forward_masked(self, x, idx_keep):
        x = self.tokenizer(x)
        x = self.encoder.forward_masked(x, idx_keep=idx_keep)
        return x

    @classmethod
    def from_timm_vit(cls, image_size: int, num_channels: int, model_name: str = 'vit_base_patch16_224'):
        """Create a backbone from a timm ViT model."""
        from ai4bmr_learn.models.encoder.timm import Tokenizer, MaskedEncoder
        from timm import create_model

        model = create_model(
            model_name=model_name,
            pretrained=False,
            num_classes=0,
            global_pool="",
            img_size=image_size,
            in_chans=num_channels
        )

        tokenizer = Tokenizer(model, image_size=image_size)
        encoder = MaskedEncoder(model, num_patches=tokenizer.num_tokens)

        return cls(tokenizer=tokenizer, encoder=encoder)
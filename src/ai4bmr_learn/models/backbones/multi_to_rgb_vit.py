from ai4bmr_learn.models.backbones.base_backbone import BaseBackbone
import torch.nn as nn
import torch

from ai4bmr_learn.models.tokenizer.base import BaseTokenizer


class MultiChannelVit(BaseBackbone):

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
            in_chans=3
        )

        conv = nn.Conv2d(in_channels=num_channels, out_channels=1, kernel_size=1, bias=True)
        tokenizer = Tokenizer(model, image_size=image_size)
        tokenizer.model = nn.Sequential(conv, tokenizer)

        encoder = MaskedEncoder(model, num_patches=tokenizer.num_tokens)

        return cls(tokenizer=tokenizer, encoder=encoder)


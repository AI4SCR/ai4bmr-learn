from ai4bmr_learn.models.backbones.base_backbone import BaseBackbone
import torch.nn as nn
import torch

from ai4bmr_learn.models.tokenizer.base import BaseTokenizer


class MultiChannelVit(BaseBackbone):

    @classmethod
    def from_timm_vit(cls,
                      image_size: int,
                      num_channels: int,
                      model_name: str = 'vit_base_patch16_224',
                      pretrained: bool = True,
                      freeze_encoder: bool = True,
                      ):
        """Create a backbone from a timm ViT model."""
        from ai4bmr_learn.models.encoder.timm import Tokenizer, MaskedEncoder
        from timm import create_model

        model = create_model(
            model_name=model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="",
            img_size=image_size,
            in_chans=3
        )

        conv = nn.Conv2d(in_channels=num_channels, out_channels=3, kernel_size=1, bias=True)
        tokenizer = tk = Tokenizer(model, image_size=image_size)
        tk.model = nn.Sequential(conv, tk.model)
        tk.num_token_pixels = tk.kernel_size[0] * tk.kernel_size[1] * num_channels
        encoder = MaskedEncoder(model, num_patches=tk.num_tokens)

        if freeze_encoder:
            for param in model.parameters():
                param.requires_grad = False
            for param in tokenizer.parameters():
                param.requires_grad = False
            for param in conv.parameters():
                param.requires_grad = True

        return cls(tokenizer=tokenizer, encoder=encoder)


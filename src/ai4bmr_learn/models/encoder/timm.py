# check: resample_abs_pos_embed in timm
import torch
from timm.models.vision_transformer import VisionTransformer
from ai4bmr_learn.models.tokenizer.base import BaseTokenizer
from ai4bmr_learn.models.encoder.masked_encoder import BaseMaskedEncoder

from ai4bmr_learn.models.utils import get_at_index


class Tokenizer(BaseTokenizer):
    def __init__(self, model, image_size: int):

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
    def __init__(self, model, num_patches: int):
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


def get_timm_backbones(
        # tokenizer
        image_size: int = 224,
        num_channels: int = 3,
        kernel_size: int = 16,
        token_dim: int = 768,
        # encoder
        encoder_layers: int = 12,
        encoder_heads: int = 3,
        # decoder
        decoder_dim: int = 768,
        decoder_layers: int = 4,
        decoder_heads: int = 3,
        # pretrained
        model_name: str = None,  # 'vit_base_patch16_224', 'vit_base_patch8_224', 'vit_relpos_base_patch16_224'
        pretrained: bool = True,
        strict: bool = True,  # this will not use the default kwargs of TimmModel when creating the model
    ):

        timm_kwargs = dict(
            img_size=image_size,
            patch_size=kernel_size,
            in_chans=num_channels,
            embed_dim=token_dim,
            depth=encoder_layers,
            num_heads=encoder_heads,
        )
        # timm.list_models('vit*')
        if model_name is not None:
            import timm

            if strict:
                print(f"Loading model '{model_name}' strictly from timm")
                model = timm.create_model(model_name, pretrained=False)
                assert image_size == get_image_size_from_name(model_name)
            else:
                print(f"Loading model '{model_name}' from timm")
                model = timm.create_model(model_name, pretrained=False, **timm_kwargs)

            if pretrained:
                from timm.models import load_pretrained

                print(f"Loading pretrained weights")
                # NOTE: we use the sequence of pretrained=False and load_pretrained to enable strict or partial loading of
                #   pretrained weights. This still does not enforce completely matching the original model architecture.
                #   For example, we can load a Vit with different num_heads from pretrained weights with strict=True
                load_pretrained(model, strict=strict)
        else:
            print(f"Create new ViT model")
            model = VisionTransformer(**timm_kwargs)

        tokenizer = Tokenizer(model=model, image_size=image_size)
        encoder = MaskedEncoder(model=model, num_patches=tokenizer.num_patches)
        decoder = Masked(
            num_tokens=encoder.num_tokens,
            dim=decoder_dim,
            num_layers=decoder_layers,
            num_heads=decoder_heads,
        )

def get_image_size_from_name(model_name: str):
    return int(model_name.split("_")[-1])

# TimmModel()

import torch
import torch.nn as nn
from einops import rearrange
from ai4bmr_learn.models.tokenizer.base import BaseTokenizer


class TokenizerLinear(BaseTokenizer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.flatten = nn.Flatten(start_dim=2)
        self.proj = nn.Linear(self.num_patch_pixels, self.dim)
        self.norm = nn.LayerNorm(self.dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        assert (
                images.shape[2:] == self.image_size
        ), f"image size mismatch, expected {self.image_size} but got {images.shape[2:]}"

        from einops import rearrange
        kh, kw = self.kernel_size
        h, w = self.grid_size
        tokens = rearrange(images, "b c (h kh) (w kw) -> b (h w) (c kh kw)", kh=kh, kw=kw, h=h, w=w)
        assert tokens.shape[1] == self.num_tokens

        x = self.flatten(tokens)
        x = self.proj(x)
        x = self.norm(x)
        return x

    def tokens2img(self, x: torch.Tensor) -> torch.Tensor:
        kh, kw = self.kernel_size
        h, w = self.grid_size
        img = rearrange(x, "b (h w) (c kh kw) -> b c (h kh) (w kw)", kh=kh, kw=kw, h=h, w=w)
        return img

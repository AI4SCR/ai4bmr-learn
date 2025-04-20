from ai4bmr_learn.utils.utils import pair
from ai4bmr_learn.models.tokenizer.utils import compute_num_tokens

from pydantic import BaseModel, ConfigDict, Field
import torch.nn as nn
import torch

class TokenizerDefault(BaseModel):
    image_size: int = 224
    kernel_size: int = 16
    dim: int = 192
    num_channels: int = 3


class TokenizerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = "conv2d"
    init_kwargs: TokenizerDefault = Field(default_factory=TokenizerDefault)


class BaseTokenizer(nn.Module):

    def __init__(self, *, image_size: int, kernel_size: int, dim: int, num_channels: int):
        super().__init__()
        self.image_size = pair(image_size)
        self.kernel_size = pair(kernel_size)
        self.stride = pair(kernel_size)

        self.num_channels = num_channels
        self.dim = dim

        self.grid_size, self.num_tokens = compute_num_tokens(
            image_size=self.image_size, kernel_size=self.kernel_size, stride=self.stride
        )
        self.num_tokens_h, self.num_tokens_w = self.grid_size
        self.num_token_pixels = self.kernel_size[0] * self.kernel_size[1] * self.num_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def tokens2img(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
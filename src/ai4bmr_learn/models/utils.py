def collect_model_stats(model):
    from torchinfo import summary

    model_summary = summary(model, verbose=0)

    # Accessing stored values
    total_params = model_summary.total_params
    trainable_params = model_summary.trainable_params
    non_trainable_params = total_params - trainable_params
    return dict(
        total_params=total_params,
        trainable_params=trainable_params,
        non_trainable_params=non_trainable_params,
    )

from pathlib import Path

import torch
from einops import repeat


def set_at_index(tokens, *, index, value):
    idx_keep_expanded = repeat(index, "b k -> b k d", d=tokens.shape[-1])
    tokens = tokens.scatter(index=idx_keep_expanded, dim=1, src=value)
    return tokens


def set_at_mask(tokens, *, mask, value):
    tokens[mask] = value.flatten(end_dim=1)
    return tokens


def get_at_index(tokens, *, index):
    idx_keep_expanded = repeat(index, "b k -> b k d", d=tokens.shape[-1])
    unmasked_tokens = torch.gather(tokens, 1, idx_keep_expanded)
    return unmasked_tokens


def get_at_mask(tokens, *, mask):
    pass


def random_token_mask(
    batch_size: int,
    num_tokens: int,
    num_prefix_tokens: int,
    mask_ratio: float = 0.75,
    mask_prefix_tokens: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # NOTE: compute the number of masked tokens from the number of tokens eligible for masking
    if mask_prefix_tokens:
        num_keep = int(num_tokens * (1 - mask_ratio))
    else:
        num_maskable_tokens = num_tokens - num_prefix_tokens
        num_keep = int(num_maskable_tokens * (1 - mask_ratio))
        num_keep += num_prefix_tokens
    num_mask = num_tokens - num_keep

    noise = torch.rand(batch_size, num_tokens)
    if not mask_prefix_tokens and num_tokens > num_prefix_tokens:
        # Make sure that prefix_tokens are not masked
        noise[:, :num_prefix_tokens] = -1

    # Get indices of tokens to keep by sorting the noise
    indices = torch.argsort(noise, dim=1)
    idx_keep = indices[:, :num_keep]
    idx_mask = indices[:, num_keep:]
    assert idx_keep.shape[-1] + idx_mask.shape[-1] == num_tokens

    token_mask = torch.zeros(batch_size, num_tokens, dtype=torch.bool)
    token_mask = torch.scatter(
        token_mask,
        dim=1,
        index=idx_mask,
        src=torch.ones_like(token_mask, dtype=torch.bool),
    )
    assert (token_mask.sum(dim=1) == num_mask).all()

    return token_mask, idx_mask, idx_keep


def get_best_checkpoint_path(ckpt_dir: Path) -> Path:
    import re

    """Find the checkpoint with lowest validation loss."""
    checkpoints = list(ckpt_dir.glob("*.ckpt"))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")

    if len(checkpoints) == 1:
        return checkpoints[0]

    def get_val_loss(ckpt: Path) -> float:
        match = re.search(r"val_loss=([\d.]+)\.", ckpt.name)
        return float(match.group(1)) if match else float("inf")

    return min(checkpoints, key=get_val_loss)

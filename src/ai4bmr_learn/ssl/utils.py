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

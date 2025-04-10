import torch


def reduce_to_rgb(tokens):
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import MinMaxScaler

    tokens = PCA(n_components=3).fit_transform(tokens)
    tokens = MinMaxScaler().fit_transform(tokens)
    tokens = torch.from_numpy(tokens)

    return tokens


from ai4bmr_core.utils.plotting import channel_to_rgba


def token_to_color(image_size: tuple[int, int], tokens, coords, alpha=0.5, cmap_name: str = "gray", patch_size=None):
    import einops
    from torch import nn
    from visualization.multi_channel_image import normalize_

    if tokens.shape[-1] > 3:
        tokens = reduce_to_rgb(tokens=tokens)

    # expand to patch_size
    tokens = einops.rearrange(tokens, "n c -> n c 1 1")
    tokens = nn.functional.interpolate(tokens, size=patch_size)

    height, width = image_size[-2:]
    attn = torch.zeros((3, height, width))
    for coord, token in zip(coords, tokens):
        top, left = coord["top"], coord["left"]
        attn[:, top : top + patch_size, left : left + patch_size] = token

    return attn


def viz_tokens(channel, tokens, coords, patch_size, alpha=0.5, cmap_name: str = "gray", grid_mode=False, rgb=None):
    # WIP: visualize the single tokens not only the summary tokens.
    import einops
    from torch import nn
    from visualization.multi_channel_image import normalize_
    from ai4bmr_core.utils.plotting import blend

    num_patches = len(tokens)
    num_tokens_per_patch = tokens.shape[1]
    num_dim = tokens.shape[-1]

    tokens = tokens.reshape(-1, num_dim)  # concat all embeddings
    tokens = reduce_to_rgb(tokens=tokens)
    tokens = tokens.reshape(num_patches, num_tokens_per_patch, -1)  # shape back to B, N, C

    num_tokens_per_dim = num_tokens_per_patch**0.5
    assert int(num_tokens_per_dim) == num_tokens_per_dim
    num_tokens_per_dim = int(num_tokens_per_dim)

    kernel_size = patch_size / num_tokens_per_dim
    assert int(kernel_size) == kernel_size
    kernel_size = int(kernel_size)

    tokens = einops.rearrange(tokens, "b n c -> b n c 1 1")
    tokens = torch.stack(
        [nn.functional.interpolate(i, size=kernel_size) for i in tokens]
    )  # note we need to process by patch

    if grid_mode:
        from torchvision.utils import make_grid

        nrow = len(set(i[0] for i in coords))
        attn = [make_grid(i, nrow=num_tokens_per_dim, normalize=False) for i in tokens]
        attn = make_grid(attn, nrow=nrow, normalize=False)
        return attn

    else:
        tokens = einops.rearrange(
            tokens,
            "b (H W) c h w -> b c (H h) (W w)",
            H=num_tokens_per_dim,
            W=num_tokens_per_dim,
        )

        height, width = channel.shape
        attn = torch.zeros((3, height, width))
        for coord, attn_ in zip(coords, tokens):
            top, left = coord
            attn[:, top : top + patch_size, left : left + patch_size] = attn_

        if rgb is None:
            rgb = normalize_(channel)
            rgb = channel_to_rgba(channel, cmap_name=cmap_name)[..., :3]

        blended = blend(channel, attn, alpha=alpha)
        return blended

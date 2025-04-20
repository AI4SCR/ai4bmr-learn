import torch
from einops import rearrange
from ai4bmr_learn.utils.utils import pair


def compute_num_tokens(*, image_size, kernel_size, stride):
    H, W = image_size
    kh, kw = pair(kernel_size)
    sh, sw = pair(stride)

    num_patches_w = (W - kw) // sw + 1
    num_patches_h = (H - kh) // sh + 1
    num_patches = num_patches_w * num_patches_h
    return (num_patches_h, num_patches_w), num_patches


def patchify(
    images,
    kernel_size: int | tuple[int, int],
    stride: int | tuple[int, int],
    fmt: str = "b (h w) c kh kw",
):
    """
    Convert an image tensor into patches using `torch.unfold`.

    Args:
        images (torch.Tensor): Input image of shape (B, C, H, W).
        kernel_size (int or tuple): Size of each square patch, either as an int or a (kh, kw) tuple.
        stride (int or tuple): Stride for the unfolding, either as an int or a (sh, sw) tuple.
        fmt (str): Format of the output patches used by einops. After unfolding the patches, the shape is
            (B, C, h, w, kh, kw), you can specify the format of the output patches using this argument.

    Returns:
        torch.Tensor: Patches in the specified format.
    """
    # Ensure the input tensor has the expected shape
    if images.ndim != 4:
        raise ValueError(
            f"Input tensor must have 4 dimensions (B, C, H, W), but got {images.shape}"
        )

    B, C, H, W = images.shape

    kh, kw = pair(kernel_size)
    sh, sw = pair(stride)

    # Check that image dimensions are compatible with kernel size and stride
    assert (
        H % kh == 0 and W % kw == 0
    ), f"Image dimensions ({H}, {W}) must be divisible by patch_size {kernel_size}"

    # Unfold into patches
    patches = images.unfold(2, kh, sh).unfold(3, kw, sw)
    # Shape: (B, C, H_unfold, W_unfold, kh, kw)

    # Move dimensions to get (B, H, W, C, kh, kw)
    patches = patches.permute(0, 2, 3, 1, 4, 5)
    # Shape: (B, H_unfold, W_unfold, C, kh, kw)

    return rearrange(patches, "b h w c kh kw -> " + fmt)


def patches2image(
    patches,
    image_size: int | tuple[int, int],
    kernel_size: int | tuple[int, int],
    stride: int | tuple[int, int],
    fmt: str = "b (h w) c kh kw",
):
    kh, kw = pair(kernel_size)
    (h, w), num_patches = compute_num_tokens(
        image_size=image_size, kernel_size=kernel_size, stride=stride
    )
    return rearrange(patches, f"{fmt} -> b c (h kh) (w kw)", kh=kh, kw=kw, h=h, w=w)
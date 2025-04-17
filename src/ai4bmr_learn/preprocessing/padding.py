from ai4bmr_learn.utils.utils import pair

def get_pad_for_kernel(height: int, width: int, kernel_size: int, stride: int) -> tuple[int, int]:
    """Compute the minimal padding of height and width to make image divisible by kernel and stride."""

    kh, kw = pair(kernel_size)
    sh, sw = pair(stride)

    if height <= kh:
        ph = kernel_size - height
    else:
        ph = sh - (height - (height // sh) * sh)

    if width <= kw:
        pw = kernel_size - width
    else:
        pw = sw - (width - (width // sw) * sw)

    return ph, pw


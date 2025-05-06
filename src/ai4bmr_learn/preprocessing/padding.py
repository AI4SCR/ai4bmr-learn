from ai4bmr_learn.utils.utils import pair

def get_pad_for_kernel(height: int, width: int, kernel_size: int, stride: int) -> tuple[int, int]:
    """Compute the minimal padding of height and width to make image divisible by kernel and stride."""

    kh, kw = pair(kernel_size)
    sh, sw = pair(stride)

    if height <= kh:
        ph = kernel_size - height
    else:
        ph = height - (height // sh) * sh
        ph = sh - ph if ph else ph

    if width <= kw:
        pw = kernel_size - width
    else:
        pw = width - (width // sw) * sw
        pw = sw - pw if pw else pw

    return ph, pw


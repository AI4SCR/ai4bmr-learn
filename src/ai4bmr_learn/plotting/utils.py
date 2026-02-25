import matplotlib.pyplot as plt
import numpy as np


def get_colorcet_map(items: list, as_int: bool = True) -> dict:
    import colorcet as cc

    unique_items = sorted(set(items))
    glasbey_colors = cc.glasbey_bw[: len(unique_items)]
    scale = 255 if as_int else 1
    return {
        item: tuple(int(scale * channel) for channel in rgb)
        for item, rgb in zip(unique_items, glasbey_colors)
    }


def get_grid_dims(n_samples: int) -> tuple[int, int]:
    assert n_samples > 0, "n_samples must be > 0"
    n_rows = int(np.ceil(np.sqrt(n_samples)))
    n_cols = int(np.ceil(n_samples / n_rows))
    return n_rows, n_cols


def channel_to_rgba(channel: np.ndarray, cmap=None, cmap_name: str | None = None) -> np.ndarray:
    color_map = cmap or plt.get_cmap(cmap_name)
    return color_map(channel)


def blend(img1: np.ndarray, img2: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    assert 0.0 <= alpha <= 1.0, "alpha must be in [0, 1]"
    return alpha * img1 + (1.0 - alpha) * img2

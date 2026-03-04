from __future__ import annotations

import numpy as np
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def _to_sample_feature_matrix(data: np.ndarray) -> np.ndarray:
    array = np.asarray(data)
    assert array.ndim == 3, "Expected CHW array."
    channels, height, width = array.shape
    return array.reshape(channels, height * width).T


class ChannelStatistics:
    """Incremental per-channel scalers for CHW images."""

    def __init__(self):
        self.standard_scaler = StandardScaler()
        self.minmax_scaler = MinMaxScaler()
        self.num_pixels_seen = 0
        self.max_height: int | None = None
        self.max_width: int | None = None

    @property
    def mean(self) -> np.ndarray:
        return self.standard_scaler.mean_

    @property
    def std(self) -> np.ndarray:
        return self.standard_scaler.scale_

    @property
    def mins(self) -> np.ndarray:
        return self.minmax_scaler.data_min_

    @property
    def maxs(self) -> np.ndarray:
        return self.minmax_scaler.data_max_

    def partial_fit(self, image: np.ndarray) -> None:
        samples = _to_sample_feature_matrix(image)
        assert samples.size > 0, "Image has no pixels."

        _, height, width = image.shape
        self.max_height = height if self.max_height is None else max(self.max_height, height)
        self.max_width = width if self.max_width is None else max(self.max_width, width)

        self.standard_scaler.partial_fit(samples)
        self.minmax_scaler.partial_fit(samples)
        self.num_pixels_seen += samples.shape[0]

    def to_dict(self) -> dict:
        return {
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "mins": self.mins.tolist(),
            "maxs": self.maxs.tolist(),
            "max_height": self.max_height,
            "max_width": self.max_width,
            "num_pixels_seen": self.num_pixels_seen,
        }

    def to_yaml(self, save_path: Path) -> Path:
        import yaml

        target = Path(save_path).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w") as handle:
            yaml.safe_dump(self.to_dict(), handle, sort_keys=False)
        return target


def model_stats(model):
    from torchinfo import summary

    model_summary = summary(model, verbose=0)
    total_params = model_summary.total_params
    trainable_params = model_summary.trainable_params
    non_trainable_params = total_params - trainable_params
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "non_trainable_params": non_trainable_params,
    }

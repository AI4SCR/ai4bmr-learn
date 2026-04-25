from __future__ import annotations

import json
from abc import abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import glom
import numpy as np
import pandas as pd
import torch
from torch.utils.data import default_collate

from ai4bmr_learn.datasets.items import Items


class BagsDataset(Items):
    name: str = "Bags"

    def __init__(
        self,
        items_path: Path,
        metadata_path: Path | None = None,
        split: str | None = None,
        transform: Callable | None = None,
        cache_dir: Path | None = None,
        drop_nan_columns: bool = False,
        embedding_key: str = "z",
        bag_id_key: str = "sample_id",
        num_workers: int = 10,
        batch_size: int = 32,
    ):
        super().__init__(
            items_path=items_path,
            metadata_path=metadata_path,
            split=split,
            transform=transform,
            cache_dir=cache_dir,
            drop_nan_columns=drop_nan_columns,
            id_key=bag_id_key,
            num_workers=num_workers,
            batch_size=batch_size,
        )
        self.embedding_key = embedding_key
        self.bag_id_key = bag_id_key

        self.grouped_items: dict[str, list[dict[str, Any]]] | None = None
        self.bag_ids: list[str] | None = None

    def setup(self) -> None:
        super().setup()
        assert self.items is not None, "items"

        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in self.items:
            assert self.bag_id_key in item, "bag_id"
            assert self.embedding_key in item, "embedding"
            bag_id = str(item[self.bag_id_key])
            grouped[bag_id].append(item)

        bag_ids = list(grouped)
        assert bag_ids, "bags"

        if self.metadata is not None:
            metadata = self.metadata.copy()
            metadata.index = metadata.index.astype(str)
            metadata = metadata.loc[~metadata.index.duplicated(keep="first")]
            missing_bags = set(bag_ids) - set(metadata.index)
            assert not missing_bags, "metadata_bags"
            self.metadata = metadata.loc[bag_ids]

        self.grouped_items = dict(grouped)
        self.bag_ids = bag_ids

    def __len__(self) -> int:
        assert self.bag_ids is not None, "setup"
        return len(self.bag_ids)

    @abstractmethod
    def __getitem__(self, idx) -> dict[str, Any]:
        assert self.bag_ids is not None and self.grouped_items is not None, "setup"
        raise NotImplementedError("Inherit from `BagsDataset` to create your own bag dataset subclass.")

    def get_bag_id(self, idx: int) -> str:
        assert self.bag_ids is not None, "setup"
        return self.bag_ids[idx]

    def get_bag_items(self, idx: int) -> list[dict[str, Any]]:
        assert self.grouped_items is not None, "setup"
        return self.grouped_items[self.get_bag_id(idx)]

    def get_bag(self, idx: int) -> torch.Tensor:
        bag_items = self.get_bag_items(idx)
        embeddings = [torch.as_tensor(item[self.embedding_key], dtype=torch.float32).reshape(-1) for item in bag_items]
        assert embeddings, "empty bag"

        embedding_dim = embeddings[0].numel()
        assert all(embedding.numel() == embedding_dim for embedding in embeddings), "embedding_dim"
        return torch.stack(embeddings, dim=0)

    def get_metadata(self, idx: int) -> dict[str, Any] | None:
        if self.metadata is None:
            return None
        return self.metadata.loc[self.get_bag_id(idx)].to_dict()


def pad_bags_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    assert batch, "batch"
    bags = [item["bag"] for item in batch]
    assert all(isinstance(bag, torch.Tensor) for bag in bags), "bag"
    assert all(bag.ndim == 2 for bag in bags), "bag_ndim"

    feature_dim = bags[0].shape[1]
    assert all(bag.shape[1] == feature_dim for bag in bags), "bag_dim"

    batch_size = len(bags)
    max_instances = max(int(bag.shape[0]) for bag in bags)
    assert max_instances > 0, "empty bag"

    padded = bags[0].new_zeros((batch_size, max_instances, feature_dim))
    mask = torch.zeros((batch_size, max_instances), dtype=torch.bool)
    for idx, bag in enumerate(bags):
        num_instances = int(bag.shape[0])
        padded[idx, :num_instances] = bag
        mask[idx, :num_instances] = True

    collated = default_collate([{key: value for key, value in item.items() if key != "bag"} for item in batch])
    collated["bag"] = padded
    collated["mask"] = mask
    return collated


def write_mil_items_from_cache(
    *,
    cache_dir: Path,
    items_path: Path,
    id_key: str,
    embedding_key: str = "z",
    bag_id_key: str = "sample_id",
) -> Path:
    cache_dir = Path(cache_dir).expanduser().resolve()
    items_path = Path(items_path).expanduser().resolve()
    assert cache_dir.exists(), "cache_dir"

    cache_files = sorted(cache_dir.glob("*.pt"))
    assert cache_files, "cache_files"

    items: list[dict[str, Any]] = []
    idx = 0
    for cache_file in cache_files:
        outputs = torch.load(cache_file, map_location="cpu")
        assert isinstance(outputs, list) and outputs, "cache_outputs"
        assert all(isinstance(output, dict) for output in outputs), "cache_output"

        for output in outputs:
            ids = _as_list(glom.glom(output, id_key))
            ids = [str(value) for value in ids]

            embeddings = torch.as_tensor(glom.glom(output, embedding_key), dtype=torch.float32)
            assert embeddings.ndim == 2, "embeddings_ndim"
            assert embeddings.shape[0] == len(ids), "embeddings_ids"
            for bag_id, embedding in zip(ids, embeddings):
                items.append(
                    {
                        "id": str(idx),
                        bag_id_key: bag_id,
                        embedding_key: embedding.tolist(),
                    }
                )
                idx += 1

    assert items, "items"
    items_path.parent.mkdir(parents=True, exist_ok=True)
    items_path.write_text(json.dumps(items), encoding="utf-8")
    return items_path


def _as_list(value: Any) -> list[Any]:
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value.reshape(-1).tolist()
    if isinstance(value, (list, tuple)):
        return np.asarray(value, dtype=object).reshape(-1).tolist()
    return [value]

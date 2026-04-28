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
from loguru import logger
from torch.utils.data import default_collate

from ai4bmr_learn.datasets.items import Items
from ai4bmr_learn.datasets.utils import filter_items_and_metadata


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
        self.embedding_path_key = f"{embedding_key}_path"

        self.grouped_items: dict[str, list[dict[str, Any]]] | None = None
        self.bag_ids: list[str] | None = None

    def setup(self) -> None:
        logger.info(f"Setting up {self.name} dataset from items_path: {self.items_path}")
        self._load_items()
        assert self.items is not None, "items"

        if self.metadata_path is not None:
            logger.info(f"Loading metadata from {self.metadata_path}")
            item_ids = [item[self.bag_id_key] for item in self.items]
            metadata = pd.read_parquet(self.metadata_path)
            self.item_ids, self.metadata = filter_items_and_metadata(
                item_ids=item_ids,
                metadata=metadata,
                split=self.split,
                drop_nan_columns=self.drop_nan_columns,
            )
            item_id_set = set(self.item_ids)
            self.items = [item for item in self.items if item[self.bag_id_key] in item_id_set]

        if self.cache_dir and not self.has_cache():
            logger.info(f"No cache found at {self.cache_dir}. Creating...")
            self.create_cache()

        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in self.items:
            assert self.bag_id_key in item, "bag_id"
            assert self.embedding_key in item or self.embedding_path_key in item, "embedding"
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
        if self.embedding_path_key in bag_items[0]:
            assert len(bag_items) == 1, "bag_item_count"
            bag_payload = torch.load(Path(bag_items[0][self.embedding_path_key]), map_location="cpu")
            assert isinstance(bag_payload, dict), "bag_payload"
            assert self.embedding_key in bag_payload, "embedding"
            embeddings = torch.as_tensor(bag_payload[self.embedding_key], dtype=torch.float32)
            assert embeddings.ndim == 2, "embeddings_ndim"
            return embeddings

        embeddings = [torch.as_tensor(item[self.embedding_key], dtype=torch.float32).reshape(-1) for item in bag_items]
        assert embeddings, "empty bag"

        embedding_dim = embeddings[0].numel()
        assert all(embedding.numel() == embedding_dim for embedding in embeddings), "embedding_dim"
        return torch.stack(embeddings, dim=0)

    def get_metadata(self, idx: int) -> dict[str, Any] | None:
        if self.metadata is None:
            return None
        return self.metadata.loc[self.get_bag_id(idx)].to_dict()

    def _load_items(self) -> None:
        with open(self.items_path, "r", encoding="utf-8") as f:
            raw_items = json.load(f)

        if isinstance(raw_items, list):
            self.items = raw_items
        else:
            assert isinstance(raw_items, dict) and raw_items, "items"
            self.items = []
            for bag_id, payload in raw_items.items():
                assert isinstance(payload, dict), "bag_payload"
                assert "instance_ids" in payload, "instance_ids"
                assert self.embedding_path_key in payload, "embedding_path"
                self.items.append(
                    {
                        self.bag_id_key: str(bag_id),
                        "instance_ids": payload["instance_ids"],
                        self.embedding_path_key: str(Path(payload[self.embedding_path_key]).expanduser().resolve()),
                    }
                )

        self.item_ids = [item[self.bag_id_key] for item in self.items]
        logger.info(f"Loaded {len(self.items)} items")


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
    assert items_path.suffix == ".json", "items_path_suffix"

    cache_files = sorted(cache_dir.glob("*.pt"))
    assert cache_files, "cache_files"

    embedding_path_key = f"{embedding_key}_path"
    bags_dir = items_path.parent / items_path.stem
    bags_dir.mkdir(parents=True, exist_ok=True)

    bag_embeddings: dict[str, list[torch.Tensor]] = defaultdict(list)
    bag_instance_ids: dict[str, list[str]] = defaultdict(list)
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
                bag_embeddings[bag_id].append(embedding.reshape(1, -1))
                bag_instance_ids[bag_id].append(str(idx))
                idx += 1

    assert bag_embeddings, "items"
    items: dict[str, dict[str, Any]] = {}
    for bag_id, embeddings_list in bag_embeddings.items():
        bag_tensor = torch.cat(embeddings_list, dim=0)
        bag_path = bags_dir / f"{bag_id}.pt"
        torch.save(
            {
                bag_id_key: bag_id,
                embedding_key: bag_tensor,
            },
            bag_path,
        )
        items[bag_id] = {
            "instance_ids": bag_instance_ids[bag_id],
            embedding_path_key: str(bag_path),
        }

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

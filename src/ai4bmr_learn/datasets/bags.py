from __future__ import annotations

import copy
import json
from abc import abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from glom import assign, delete, glom
from loguru import logger
from torch.utils.data import default_collate

from ai4bmr_learn.datasets.items import Items


class BagsDataset(Items):
    name: str = "Bags"

    def __init__(self, *args, bag_id_key: str, strict: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.bag_id_key = bag_id_key
        self.strict = strict
        self.bag_items: dict[Any, list[dict[str, Any]]] | None = None
        self.bag_ids: list[Any] | None = None

    def __len__(self) -> int:
        assert self.bag_ids is not None, "setup"
        return len(self.bag_ids)

    @abstractmethod
    def __getitem__(self, idx) -> dict[str, Any]:
        assert self.bag_ids is not None and self.bag_items is not None, "setup"
        raise NotImplementedError("Inherit from `BagsDataset` to create your own bag dataset subclass.")

    def setup(self):
        super().setup()
        self.build_bag_items()
        self.build_bag_metadata()

    def build_bag_items(self) -> None:
        assert self.items is not None, "setup"
        bag_items: dict[Any, list[dict[str, Any]]] = {}
        for item in self.items:
            bag_id = glom(item, self.bag_id_key)
            bag_items.setdefault(bag_id, []).append(item)

        self.bag_items = bag_items
        self.bag_ids = list(bag_items)

    def build_bag_metadata(self) -> None:
        assert self.bag_ids is not None and self.bag_items is not None, "setup"
        if self.metadata is None:
            return

        assert self.id_key is not None, "id_key"
        assert self.item_ids is not None, "setup"
        assert len(self.items) == len(self.item_ids), "items"

        bag_ids_by_item_id = {
            item_id: glom(item, self.bag_id_key)
            for item_id, item in zip(self.item_ids, self.items, strict=True)
        }
        bag_rows = {
            bag_id: self.metadata.loc[[item_id for item_id, mapped_bag_id in bag_ids_by_item_id.items() if mapped_bag_id == bag_id]]
            for bag_id in self.bag_ids
        }

        keep_columns: list[str] = []
        dropped_columns: list[str] = []
        for column in self.metadata.columns:
            inconsistent = any(rows[column].nunique(dropna=False) > 1 for rows in bag_rows.values())
            if inconsistent:
                if self.strict:
                    raise AssertionError(f"inconsistent bag metadata for {column}")
                dropped_columns.append(column)
                continue
            keep_columns.append(column)

        if dropped_columns:
            logger.warning(
                "Detected bag metadata columns with inconsistent values across instances in the same bag. "
                f"Dropping them. Affected columns: {dropped_columns}."
            )

        rows = []
        for bag_id in self.bag_ids:
            row = {column: bag_rows[bag_id].iloc[0][column] for column in keep_columns}
            rows.append(row)

        self.metadata = pd.DataFrame(rows, index=pd.Index(self.bag_ids, name=self.bag_id_key))


def pad_bags_collate(batch: list[dict[str, Any]], bag_key: str = "bag") -> dict[str, Any]:
    assert batch, "batch"
    bags = [glom(item, bag_key) for item in batch]
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

    stripped_batch = []
    for item in batch:
        stripped_item = copy.deepcopy(item)
        delete(stripped_item, bag_key)
        stripped_batch.append(stripped_item)

    collated = default_collate(stripped_batch)
    assign(collated, bag_key, padded, missing=dict)
    collated["mask"] = mask
    return collated


def write_mil_items_from_cache(
    *,
    cache_dir: Path,
    output_dir: Path,
    id_key: str,
    embedding_key: str = "z",
    bag_id_key: str = "sample_id",
) -> Path:
    cache_dir = Path(cache_dir).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    assert cache_dir.exists(), "cache_dir"

    cache_files = sorted(cache_dir.glob("*.pt"))
    assert cache_files, "cache_files"

    embedding_path_key = f"{embedding_key}_path"
    output_dir.mkdir(parents=True, exist_ok=True)
    items_path = output_dir / "bags.json"

    bag_embeddings: dict[Any, list[torch.Tensor]] = defaultdict(list)
    bag_instance_ids: dict[str, list[str]] = defaultdict(list)
    idx = 0
    for cache_file in cache_files:
        outputs = torch.load(cache_file, map_location="cpu")
        assert isinstance(outputs, list) and outputs, "cache_outputs"
        assert all(isinstance(output, dict) for output in outputs), "cache_output"

        for output in outputs:
            ids = _as_list(glom(output, id_key))

            embeddings = torch.as_tensor(glom(output, embedding_key), dtype=torch.float32)
            assert embeddings.ndim == 2, "embeddings_ndim"
            assert embeddings.shape[0] == len(ids), "embeddings_ids"
            for bag_id, embedding in zip(ids, embeddings):
                bag_embeddings[bag_id].append(embedding.reshape(1, -1))
                bag_instance_ids[bag_id].append(str(idx))
                idx += 1

    assert bag_embeddings, "items"
    items: list[dict[str, Any]] = []
    for bag_id, embeddings_list in bag_embeddings.items():
        bag_tensor = torch.cat(embeddings_list, dim=0)
        bag_path = output_dir / f"{bag_id}.pt"
        torch.save(
            {
                bag_id_key: bag_id,
                embedding_key: bag_tensor,
            },
            bag_path,
        )
        items.append(
            {
                bag_id_key: bag_id,
                "instance_ids": bag_instance_ids[bag_id],
                embedding_path_key: str(bag_path),
            }
        )

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

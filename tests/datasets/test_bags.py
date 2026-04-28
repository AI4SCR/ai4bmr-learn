import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from ai4bmr_learn.datasets import BagsDataset, pad_bags_collate, write_mil_items_from_cache


class DummyBags(BagsDataset):
    def __init__(self, *args, target_key: str | None = None, target_dtype: torch.dtype | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.target_key = target_key
        self.target_dtype = target_dtype

    def __getitem__(self, idx) -> dict:
        item = {
            "bag_id": self.get_bag_id(idx),
            "bag": self.get_bag(idx),
        }
        metadata = self.get_metadata(idx)
        if metadata is not None:
            item["metadata"] = metadata
            if self.target_key is not None:
                assert self.target_key in metadata, "target"
                target = metadata[self.target_key]
                assert pd.notna(target), "target_nan"
                if self.target_dtype is not None:
                    target = torch.as_tensor(target, dtype=self.target_dtype)
                item["target"] = target

        if self.transform is not None:
            item = self.transform(item)
        return item

    def get_targets(self) -> np.ndarray:
        assert self.bag_ids is not None and self.metadata is not None, "setup"
        assert self.target_key is not None and self.target_dtype is not None, "target"
        if self.target_dtype == torch.long:
            return self.metadata.loc[self.bag_ids, self.target_key].to_numpy(dtype=np.int64)
        return self.metadata.loc[self.bag_ids, self.target_key].to_numpy(dtype=np.float32)


def write_bag_items(tmp_path: Path) -> Path:
    bags_dir = tmp_path / "items"
    bags_dir.mkdir()
    torch.save({"sample_id": "b", "z": torch.tensor([[1.0, 2.0], [5.0, 6.0]])}, bags_dir / "b.pt")
    torch.save({"sample_id": "a", "z": torch.tensor([[3.0, 4.0]])}, bags_dir / "a.pt")
    torch.save({"sample_id": "c", "z": torch.tensor([[7.0, 8.0]])}, bags_dir / "c.pt")
    items = [
        {"sample_id": "b", "instance_ids": ["0", "2"], "z_path": str(bags_dir / "b.pt")},
        {"sample_id": "a", "instance_ids": ["1"], "z_path": str(bags_dir / "a.pt")},
        {"sample_id": "c", "instance_ids": ["3"], "z_path": str(bags_dir / "c.pt")},
    ]
    items_path = tmp_path / "items.json"
    items_path.write_text(json.dumps(items), encoding="utf-8")
    return items_path


def write_bag_metadata(tmp_path: Path, rows: list[dict]) -> Path:
    metadata_path = tmp_path / "metadata.parquet"
    pd.DataFrame(rows).set_index("sample_id").to_parquet(metadata_path)
    return metadata_path


def write_nested_bag_items(tmp_path: Path) -> Path:
    bags_dir = tmp_path / "nested_items"
    bags_dir.mkdir()
    torch.save({"meta": {"sample_id": "b"}, "payload": {"z": torch.tensor([[1.0, 2.0], [5.0, 6.0]])}}, bags_dir / "b.pt")
    torch.save({"meta": {"sample_id": "a"}, "payload": {"z": torch.tensor([[3.0, 4.0]])}}, bags_dir / "a.pt")
    items = [
        {"bag": {"sample_id": "b"}, "instance_ids": ["0", "2"], "payload": {"z_path": str(bags_dir / "b.pt")}},
        {"bag": {"sample_id": "a"}, "instance_ids": ["1"], "payload": {"z_path": str(bags_dir / "a.pt")}},
    ]
    items_path = tmp_path / "nested_items.json"
    items_path.write_text(json.dumps(items), encoding="utf-8")
    return items_path


def test_bags_dataset_groups_filtered_items_and_exposes_metadata(tmp_path: Path):
    items_path = write_bag_items(tmp_path)
    metadata_path = write_bag_metadata(
        tmp_path,
        [
            {"sample_id": "a", "split": "fit", "target": 1, "site": "x"},
            {"sample_id": "b", "split": "fit", "target": 0, "site": "y"},
            {"sample_id": "c", "split": "test", "target": 1, "site": "z"},
        ],
    )

    dataset = DummyBags(
        items_path=items_path,
        metadata_path=metadata_path,
        split="fit",
        target_key="target",
        target_dtype=torch.long,
    )
    dataset.setup()

    assert dataset.bag_ids == ["b", "a"]
    assert len(dataset) == 2
    assert dataset.get_targets().dtype == torch.empty((), dtype=torch.long).numpy().dtype

    first = dataset[0]
    assert first["bag_id"] == "b"
    assert first["bag"].shape == (2, 2)
    assert first["target"].dtype == torch.long
    assert first["target"].item() == 0
    assert first["metadata"]["site"] == "y"


def test_bags_dataset_subclass_can_cast_targets_and_apply_transform(tmp_path: Path):
    items_path = write_bag_items(tmp_path)
    metadata_path = write_bag_metadata(
        tmp_path,
        [
            {"sample_id": "a", "target": 1.5},
            {"sample_id": "b", "target": 2.5},
            {"sample_id": "c", "target": 3.5},
        ],
    )

    def transform(item):
        item["bag"] = item["bag"] + 1
        item["transformed"] = True
        return item

    dataset = DummyBags(
        items_path=items_path,
        metadata_path=metadata_path,
        target_key="target",
        target_dtype=torch.float32,
        transform=transform,
    )
    dataset.setup()

    item = dataset[0]

    assert item["target"].dtype == torch.float32
    assert item["transformed"] is True
    assert torch.allclose(item["bag"][0], torch.tensor([2.0, 3.0]))
    assert dataset.get_targets().dtype == torch.empty((), dtype=torch.float32).numpy().dtype


def test_bags_dataset_subclass_can_reject_missing_targets(tmp_path: Path):
    items_path = write_bag_items(tmp_path)
    metadata_path = write_bag_metadata(
        tmp_path,
        [
            {"sample_id": "a", "target": 1.0},
            {"sample_id": "b", "target": None},
            {"sample_id": "c", "target": 3.0},
        ],
    )
    dataset = DummyBags(
        items_path=items_path,
        metadata_path=metadata_path,
        target_key="target",
        target_dtype=torch.float32,
    )
    dataset.setup()

    with pytest.raises(AssertionError, match="target_nan"):
        dataset[0]


def test_bags_dataset_reads_nested_bag_index_fields_with_glom(tmp_path: Path):
    items_path = write_nested_bag_items(tmp_path)
    metadata_path = write_bag_metadata(
        tmp_path,
        [
            {"sample_id": "a", "target": 1},
            {"sample_id": "b", "target": 0},
        ],
    )

    dataset = DummyBags(
        items_path=items_path,
        metadata_path=metadata_path,
        bag_id_key="bag.sample_id",
        embedding_key="payload.z",
        target_key="target",
        target_dtype=torch.long,
    )
    dataset.setup()

    assert dataset.bag_ids == ["b", "a"]
    assert torch.allclose(dataset[0]["bag"], torch.tensor([[1.0, 2.0], [5.0, 6.0]]))


def test_pad_bags_collate_pads_masks_and_collates_metadata():
    batch = [
        {
            "bag_id": "a",
            "bag": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            "target": torch.tensor(1),
            "metadata": {"site": "x", "score": 2.0},
        },
        {
            "bag_id": "b",
            "bag": torch.tensor([[5.0, 6.0]]),
            "target": torch.tensor(0),
            "metadata": {"site": "y", "score": 3.0},
        },
    ]

    collated = pad_bags_collate(batch)

    assert collated["bag"].shape == (2, 2, 2)
    assert torch.equal(collated["mask"], torch.tensor([[True, True], [True, False]]))
    assert torch.equal(collated["target"], torch.tensor([1, 0]))
    assert collated["bag_id"] == ["a", "b"]
    assert collated["metadata"]["site"] == ["x", "y"]
    assert torch.allclose(collated["metadata"]["score"].float(), torch.tensor([2.0, 3.0]))
    assert torch.equal(collated["bag"][1, 1], torch.zeros(2))


def test_write_mil_items_from_cache_writes_bags_dataset_compatible_items(tmp_path: Path):
    cache_dir = tmp_path / "cache" / "prediction"
    cache_dir.mkdir(parents=True)
    torch.save(
        [
            {
                "nested": {"sample_id": ["b", "a"]},
                "z": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            }
        ],
        cache_dir / "000001.pt",
    )
    torch.save(
        [
            {
                "nested": {"sample_id": ["b"]},
                "z": torch.tensor([[5.0, 6.0]]),
            }
        ],
        cache_dir / "000002.pt",
    )

    output_dir = tmp_path / "mil_items"
    items_path = write_mil_items_from_cache(
        cache_dir=cache_dir,
        output_dir=output_dir,
        id_key="nested.sample_id",
    )

    assert items_path == output_dir / "bags.json"
    items = json.loads(items_path.read_text(encoding="utf-8"))
    assert [item["sample_id"] for item in items] == ["b", "a"]
    assert [item["instance_ids"] for item in items] == [["0", "2"], ["1"]]
    assert Path(items[0]["z_path"]).parent == output_dir
    assert Path(items[1]["z_path"]).parent == output_dir

    a_payload = torch.load(items[0]["z_path"], map_location="cpu")
    b_payload = torch.load(items[1]["z_path"], map_location="cpu")
    assert a_payload["sample_id"] == "b"
    assert b_payload["sample_id"] == "a"
    assert torch.allclose(a_payload["z"], torch.tensor([[1.0, 2.0], [5.0, 6.0]]))
    assert torch.allclose(b_payload["z"], torch.tensor([[3.0, 4.0]]))

    metadata_path = write_bag_metadata(
        tmp_path,
        [
            {"sample_id": "a", "target": 1},
            {"sample_id": "b", "target": 0},
        ],
    )
    dataset = DummyBags(
        items_path=items_path,
        metadata_path=metadata_path,
        target_key="target",
        target_dtype=torch.long,
    )
    dataset.setup()

    assert dataset.bag_ids == ["b", "a"]
    assert dataset[0]["bag"].shape == (2, 2)

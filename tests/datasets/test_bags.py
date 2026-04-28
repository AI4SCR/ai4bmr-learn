import io
import json
from pathlib import Path

import pandas as pd
import pytest
import torch
from loguru import logger

from ai4bmr_learn.datasets import BagsDataset, pad_bags_collate, write_mil_items_from_cache


class DummyBags(BagsDataset):
    def __init__(self, *args, bag_tensor_key: str = "bag", target_key: str | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.bag_tensor_key = bag_tensor_key
        self.target_key = target_key

    def __getitem__(self, idx) -> dict:
        assert self.bag_ids is not None and self.bag_items is not None, "setup"
        bag_id = self.bag_ids[idx]
        bag_items = self.bag_items[bag_id]
        bag = torch.stack([torch.as_tensor(item["embedding"], dtype=torch.float32) for item in bag_items])
        item = {
            "bag_id": bag_id,
            self.bag_tensor_key: bag,
        }
        if self.metadata is not None:
            metadata = self.metadata.loc[bag_id].to_dict()
            item["metadata"] = metadata
            if self.target_key is not None:
                item["target"] = torch.as_tensor(metadata[self.target_key], dtype=torch.long)

        if self.transform is not None:
            item = self.transform(item)
        return item


def write_item_rows(tmp_path: Path) -> Path:
    items = [
        {"item_id": "0", "bag_id": "b", "embedding": [1.0, 2.0]},
        {"item_id": "1", "bag_id": "a", "embedding": [3.0, 4.0]},
        {"item_id": "2", "bag_id": "b", "embedding": [5.0, 6.0]},
        {"item_id": "3", "bag_id": "c", "embedding": [7.0, 8.0]},
    ]
    items_path = tmp_path / "items.json"
    items_path.write_text(json.dumps(items), encoding="utf-8")
    return items_path


def write_nested_item_rows(tmp_path: Path) -> Path:
    items = [
        {"item_id": "0", "bag": {"sample_id": "b"}, "payload": {"embedding": [1.0, 2.0]}},
        {"item_id": "1", "bag": {"sample_id": "a"}, "payload": {"embedding": [3.0, 4.0]}},
        {"item_id": "2", "bag": {"sample_id": "b"}, "payload": {"embedding": [5.0, 6.0]}},
    ]
    items_path = tmp_path / "nested_items.json"
    items_path.write_text(json.dumps(items), encoding="utf-8")
    return items_path


def write_item_metadata(tmp_path: Path, rows: list[dict]) -> Path:
    metadata_path = tmp_path / "metadata.parquet"
    pd.DataFrame(rows).set_index("item_id").to_parquet(metadata_path)
    return metadata_path


def test_bags_dataset_groups_filtered_items_and_reduces_metadata(tmp_path: Path):
    items_path = write_item_rows(tmp_path)
    metadata_path = write_item_metadata(
        tmp_path,
        [
            {"item_id": "0", "split": "fit", "target": 0, "site": "y"},
            {"item_id": "1", "split": "fit", "target": 1, "site": "x"},
            {"item_id": "2", "split": "fit", "target": 0, "site": "y"},
            {"item_id": "3", "split": "test", "target": 1, "site": "z"},
        ],
    )

    dataset = DummyBags(
        items_path=items_path,
        metadata_path=metadata_path,
        id_key="item_id",
        bag_id_key="bag_id",
        split="fit",
        target_key="target",
    )
    dataset.setup()

    assert dataset.bag_ids == ["b", "a"]
    assert len(dataset) == 2
    assert list(dataset.bag_items) == ["b", "a"]
    assert [item["item_id"] for item in dataset.bag_items["b"]] == ["0", "2"]
    assert list(dataset.metadata.index) == ["b", "a"]
    assert dataset.metadata.loc["b", "site"] == "y"

    first = dataset[0]
    assert first["bag_id"] == "b"
    assert first["bag"].shape == (2, 2)
    assert first["target"].dtype == torch.long
    assert first["target"].item() == 0
    assert first["metadata"]["site"] == "y"


def test_bags_dataset_subclass_can_apply_transform(tmp_path: Path):
    items_path = write_item_rows(tmp_path)
    metadata_path = write_item_metadata(
        tmp_path,
        [
            {"item_id": "0", "target": 0},
            {"item_id": "1", "target": 1},
            {"item_id": "2", "target": 0},
            {"item_id": "3", "target": 1},
        ],
    )

    def transform(item):
        item["bag"] = item["bag"] + 1
        item["transformed"] = True
        return item

    dataset = DummyBags(
        items_path=items_path,
        metadata_path=metadata_path,
        id_key="item_id",
        bag_id_key="bag_id",
        transform=transform,
    )
    dataset.setup()

    item = dataset[0]

    assert item["transformed"] is True
    assert torch.allclose(item["bag"][0], torch.tensor([2.0, 3.0]))


def test_bags_dataset_reads_nested_bag_index_fields_with_glom(tmp_path: Path):
    items_path = write_nested_item_rows(tmp_path)
    metadata_path = write_item_metadata(
        tmp_path,
        [
            {"item_id": "0", "target": 0},
            {"item_id": "1", "target": 1},
            {"item_id": "2", "target": 0},
        ],
    )

    dataset = DummyBags(
        items_path=items_path,
        metadata_path=metadata_path,
        id_key="item_id",
        bag_id_key="bag.sample_id",
    )
    dataset.setup()

    assert dataset.bag_ids == ["b", "a"]
    assert [item["item_id"] for item in dataset.bag_items["b"]] == ["0", "2"]


def test_bags_dataset_strict_metadata_reduction_rejects_inconsistent_columns(tmp_path: Path):
    items_path = write_item_rows(tmp_path)
    metadata_path = write_item_metadata(
        tmp_path,
        [
            {"item_id": "0", "target": 0, "site": "x"},
            {"item_id": "1", "target": 1, "site": "x"},
            {"item_id": "2", "target": 1, "site": "y"},
            {"item_id": "3", "target": 1, "site": "z"},
        ],
    )

    dataset = DummyBags(
        items_path=items_path,
        metadata_path=metadata_path,
        id_key="item_id",
        bag_id_key="bag_id",
        strict=True,
    )

    with pytest.raises(AssertionError, match="inconsistent bag metadata for target"):
        dataset.setup()


def test_bags_dataset_non_strict_metadata_reduction_drops_inconsistent_columns(tmp_path: Path):
    items_path = write_item_rows(tmp_path)
    metadata_path = write_item_metadata(
        tmp_path,
        [
            {"item_id": "0", "target": 0, "site": "y", "cohort": "fit"},
            {"item_id": "1", "target": 1, "site": "x", "cohort": "fit"},
            {"item_id": "2", "target": 1, "site": "y", "cohort": "fit"},
            {"item_id": "3", "target": 1, "site": "z", "cohort": "test"},
        ],
    )

    stream = io.StringIO()
    handler_id = logger.add(stream)
    try:
        dataset = DummyBags(
            items_path=items_path,
            metadata_path=metadata_path,
            id_key="item_id",
            bag_id_key="bag_id",
            strict=False,
        )
        dataset.setup()
    finally:
        logger.remove(handler_id)

    assert list(dataset.metadata.columns) == ["site", "cohort"]
    assert "Dropping them. Affected columns: ['target']." in stream.getvalue()


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

    collated = pad_bags_collate(batch, bag_key="bag")

    assert collated["bag"].shape == (2, 2, 2)
    assert torch.equal(collated["mask"], torch.tensor([[True, True], [True, False]]))
    assert torch.equal(collated["target"], torch.tensor([1, 0]))
    assert collated["bag_id"] == ["a", "b"]
    assert collated["metadata"]["site"] == ["x", "y"]
    assert torch.allclose(collated["metadata"]["score"].float(), torch.tensor([2.0, 3.0]))
    assert torch.equal(collated["bag"][1, 1], torch.zeros(2))


def test_pad_bags_collate_supports_nested_bag_key():
    batch = [
        {
            "bag_id": "a",
            "payload": {"bag": torch.tensor([[1.0, 2.0], [3.0, 4.0]])},
            "target": torch.tensor(1),
        },
        {
            "bag_id": "b",
            "payload": {"bag": torch.tensor([[5.0, 6.0]])},
            "target": torch.tensor(0),
        },
    ]

    collated = pad_bags_collate(batch, bag_key="payload.bag")

    assert collated["payload"]["bag"].shape == (2, 2, 2)
    assert torch.equal(collated["mask"], torch.tensor([[True, True], [True, False]]))
    assert torch.equal(collated["payload"]["bag"][1, 1], torch.zeros(2))


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

    b_payload = torch.load(items[0]["z_path"], map_location="cpu")
    a_payload = torch.load(items[1]["z_path"], map_location="cpu")
    assert b_payload["sample_id"] == "b"
    assert a_payload["sample_id"] == "a"
    assert torch.allclose(b_payload["z"], torch.tensor([[1.0, 2.0], [5.0, 6.0]]))
    assert torch.allclose(a_payload["z"], torch.tensor([[3.0, 4.0]]))

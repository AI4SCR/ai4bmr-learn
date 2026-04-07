import json

import pandas as pd
import pytest

from ai4bmr_learn.datasets import Items


class DummyItems(Items):
    def __getitem__(self, idx) -> dict:
        return self.items[idx]


def write_items(tmp_path, item_ids: list[str]):
    items_path = tmp_path / "items.json"
    items = [{"id": item_id, "value": index} for index, item_id in enumerate(item_ids)]
    items_path.write_text(json.dumps(items))
    return items_path


def write_metadata(tmp_path, rows: list[dict]):
    metadata_path = tmp_path / "metadata.parquet"
    metadata = pd.DataFrame(rows).set_index("id")
    metadata.to_parquet(metadata_path)
    return metadata_path


@pytest.mark.parametrize(
    ("metadata_rows", "split", "expected_item_ids"),
    [
        pytest.param(
            [
                {"id": "b", "split": "test", "target": 1},
                {"id": "d", "split": "fit", "target": 2},
                {"id": "a", "split": "fit", "target": 3},
                {"id": "c", "split": "val", "target": 4},
            ],
            "fit",
            ["d", "a"],
            id="all-have-metadata-fit",
        ),
        pytest.param(
            [
                {"id": "b", "split": "test", "target": 1},
                {"id": "d", "split": "fit", "target": 2},
                {"id": "a", "split": "fit", "target": 3},
                {"id": "c", "split": "val", "target": 4},
            ],
            "test",
            ["b"],
            id="all-have-metadata-test",
        ),
        pytest.param(
            [
                {"id": "b", "split": "test", "target": 1},
                {"id": "d", "split": "fit", "target": 2},
                {"id": "a", "split": "fit", "target": 3},
                {"id": "c", "split": "val", "target": 4},
            ],
            "val",
            ["c"],
            id="all-have-metadata-val",
        ),
        pytest.param(
            [
                {"id": "b", "split": "test", "target": 1},
                {"id": "d", "split": "val", "target": 2},
                {"id": "a", "split": "fit", "target": 3},
            ],
            "fit",
            ["a"],
            id="some-have-metadata-fit",
        ),
        pytest.param(
            [
                {"id": "b", "split": "test", "target": 1},
                {"id": "d", "split": "val", "target": 2},
                {"id": "a", "split": "fit", "target": 3},
            ],
            "test",
            ["b"],
            id="some-have-metadata-test",
        ),
        pytest.param(
            [
                {"id": "b", "split": "test", "target": 1},
                {"id": "d", "split": "val", "target": 2},
                {"id": "a", "split": "fit", "target": 3},
            ],
            "val",
            ["d"],
            id="some-have-metadata-val",
        ),
        pytest.param(
            [
                {"id": "x", "split": "fit", "target": 1},
                {"id": "y", "split": "test", "target": 2},
                {"id": "z", "split": "val", "target": 3},
            ],
            "fit",
            [],
            id="none-have-metadata-fit",
        ),
        pytest.param(
            [
                {"id": "x", "split": "fit", "target": 1},
                {"id": "y", "split": "test", "target": 2},
                {"id": "z", "split": "val", "target": 3},
            ],
            "test",
            [],
            id="none-have-metadata-test",
        ),
        pytest.param(
            [
                {"id": "x", "split": "fit", "target": 1},
                {"id": "y", "split": "test", "target": 2},
                {"id": "z", "split": "val", "target": 3},
            ],
            "val",
            [],
            id="none-have-metadata-val",
        ),
    ],
)
def test_items_setup_filters_by_split_and_preserves_order(tmp_path, metadata_rows, split, expected_item_ids):
    item_ids = ["d", "a", "c", "b"]
    items_path = write_items(tmp_path=tmp_path, item_ids=item_ids)
    metadata_path = write_metadata(tmp_path=tmp_path, rows=metadata_rows)

    dataset = DummyItems(
        items_path=items_path,
        metadata_path=metadata_path,
        split=split,
        id_key="id",
        drop_nan_columns=True,
    )
    dataset.setup()

    assert dataset.item_ids == expected_item_ids
    assert [item["id"] for item in dataset.items] == expected_item_ids
    assert dataset.metadata.index.tolist() == expected_item_ids


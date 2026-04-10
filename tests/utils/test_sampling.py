import pandas as pd

from ai4bmr_learn.utils.sampling import sample_min_per_group_then_uniform


def test_sample_min_per_group_then_uniform_respects_minimum_per_group():
    frame = pd.DataFrame(
        {
            "group": ["a"] * 5 + ["b"] * 10 + ["c"] * 20,
            "value": range(35),
        }
    )

    sampled = sample_min_per_group_then_uniform(frame.groupby("group"), n=12, min_per_group=3, random_state=0)

    counts = sampled.groupby("group").size()
    assert len(sampled) == 10
    assert counts.to_dict() == {"a": 3, "b": 3, "c": 4}


def test_sample_min_per_group_then_uniform_returns_full_frame_when_n_is_large():
    frame = pd.DataFrame(
        {
            "group": ["a"] * 2 + ["b"] * 3,
            "value": range(5),
        }
    )

    sampled = sample_min_per_group_then_uniform(frame.groupby("group"), n=10, random_state=0)

    pd.testing.assert_frame_equal(sampled, frame)

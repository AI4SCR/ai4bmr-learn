import pandas as pd


def sample_min_per_group_then_uniform(
        grouped: pd.core.groupby.DataFrameGroupBy,
        n: int = 10_000,
        min_per_group: int | None = None,
        random_state: int | None = None,
) -> pd.DataFrame:
    total_samples = int(grouped.size().sum())
    if n >= total_samples:
        return grouped.obj

    if min_per_group is None:
        min_per_group = n // grouped.ngroups
    else:
        assert min_per_group <= int(grouped.size().min()), "min_per_group exceeds smallest group"

    group_sizes = grouped.size()
    smaller_groups = group_sizes[group_sizes <= min_per_group]
    larger_groups = group_sizes[group_sizes > min_per_group]

    fixed_samples = pd.Series(min_per_group, index=larger_groups.index, dtype=int)
    remaining = n - int(smaller_groups.sum() + fixed_samples.sum())

    extra_capacity = larger_groups - fixed_samples
    if remaining > 0:
        extra_samples = (remaining * extra_capacity / extra_capacity.sum()).astype(int)
    else:
        extra_samples = pd.Series(0, index=extra_capacity.index, dtype=int)

    samples_per_group = pd.concat((smaller_groups, fixed_samples + extra_samples))
    assert len(samples_per_group) == grouped.ngroups, "missing group sample counts"
    assert not samples_per_group.index.duplicated().any(), "duplicate group labels"

    aligned_group_sizes, aligned_samples = group_sizes.align(samples_per_group)
    assert (aligned_group_sizes >= aligned_samples).all(), "sample count exceeds group size"
    assert int(aligned_samples.sum()) <= n, "sampled more rows than requested"

    sampled_groups = [
        grouped.get_group(group_name).sample(num_samples, random_state=random_state)
        for group_name, num_samples in aligned_samples.items()
    ]
    sampled = pd.concat(sampled_groups)

    assert not sampled.index.duplicated().any(), "sampled duplicate rows"
    return sampled

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box

from ai4bmr_learn.data_models.Coordinate import PatchCoordinate
from ai4bmr_learn.utils.images import coords_to_bboxs, get_coordinates_dict
from ai4bmr_learn.utils.helpers import pair


def generate_points_subsets(
    *,
    points: gpd.GeoDataFrame,
    coords: list[PatchCoordinate],
    col_name: str,
    predicate: str = 'intersects',
    allow_duplicates: bool = False,
    remove_duplicates: bool = True,
):
    """
    Assigns points transcript coordinates to spatial patches based on a spatial join.

    Args:
        points (gpd.GeoDataFrame): GeoDataFrame containing transcript point geometries.
        coords (list[PatchCoordinate]): List of coordinate objects defining patch regions (via `coord_to_bbox`).
        col_name (str): Name of the column to store the patch ID assignment in the resulting DataFrame.
        predicate (str, optional): Spatial predicate for the join (e.g., 'within' or 'intersects'). Default is 'intersects'.
        allow_duplicates (bool, optional): If False, raises an error when transcript IDs are assigned to multiple patches. Default is False.
        remove_duplicates (bool, optional): If True, removes duplicated transcript IDs after the join (e.g., if overlapping patches). Default is True.

    Returns:
        gpd.GeoDataFrame: A modified copy of `points` with a new column (`col_name`) containing patch assignments.

    Notes:
        - Using 'intersects' includes transcripts exactly on patch boundaries; 'within' does not.
        - Setting `remove_duplicates=True` is useful when patches may overlap.
        - If `allow_duplicates=False` and duplicates remain after join, an `AssertionError` is raised.
    """

    bboxs = coords_to_bboxs(coords)
    patch_gdf = gpd.GeoDataFrame(geometry=bboxs, crs=points.crs)
    points = gpd.sjoin(points, patch_gdf, predicate=predicate)
    points.rename(columns={'index_right': col_name}, inplace=True)

    if not allow_duplicates:
        assert not points.transcript_id.duplicated().any()

    if remove_duplicates:
        filter_ = points.transcript_id.duplicated()
        points = points[~filter_]

    return points


def point_to_bbox(*, x: int, y: int, kernel_size: int):
    height, width = pair(kernel_size)
    x_min, y_min = x, y
    x_max, y_max = x + width, y + height

    # Create a shapely "box" polygon
    return box(x_min, y_min, x_max, y_max)


def compute_points_tokens(
    points: gpd.GeoDataFrame,
    patch_size: tuple[int, int] | int,
    kernel_size: int,
    stride: int,
    group_by: str = 'ensembl_id',
) -> pd.DataFrame:
    """Computes token-level expression counts by aggregating expression data within sliding windows.

    Args:
        points: GeoDataFrame containing transcript-level expression data for a single image patch.
        patch_size: Size of the patch (height, width) or a single int if square.
        kernel_size: Size of each token (sub-region) within the patch.
        stride: Stride used to slide the kernel across the patch.
        group_by: Column name used to group expression counts (e.g., 'ensembl_id').

    Returns:
        GeoDataFrame with expression counts aggregated at the token level,
        indexed by token and grouped by `group_by`.
    """
    patch_height, patch_width = pair(patch_size)

    # create token coordinates
    token_coords_dicts = get_coordinates_dict(
        height=patch_height,
        width=patch_width,
        kernel_size=kernel_size,
        stride=stride,
    )
    assert len(token_coords_dicts) == (patch_height / kernel_size) * (patch_width / kernel_size)
    token_coords = [PatchCoordinate(**i) for i in token_coords_dicts]

    # compute expression subsets per token
    subsets = generate_points_subsets(
        points=points,
        coords=token_coords,
        col_name='token_index',
        predicate='intersects',
        allow_duplicates=True,
        remove_duplicates=True,
    )
    # note: all transcripts must remain after assigning them to the tokens
    assert len(subsets) == len(points)

    # note: we need categorical to be able to harmonize the num_genes within a batch to be stacked
    subsets[group_by] = pd.Categorical(subsets[group_by])
    tokens = points_pool(subsets, num_tokens=len(token_coords), group_by=group_by)
    assert not tokens.isna().any().any(), 'Pooled points data contains NaN values after pooling.'
    return tokens


def points_pool(
    points_subsets: pd.DataFrame, pooling: str = 'cnt', num_tokens: int | None = None, group_by: str = 'ensembl_id'
) -> pd.DataFrame:
    """Pool points data to the token level.
    Args:
        points_subsets: all transcripts for one patch
        pooling: pooling strategy
        num_tokens: the number of tokens to return
        group_by: the column to accumulate the counts by
    Returns:
        token level expression data: dataframe with token_index as index and group_by as columns
    """
    assert points_subsets[group_by].dtype == 'category', f'{group_by} must be categorical'

    if len(points_subsets) == 0:
        # NOTE: this covers the case where there are no transcripts in a patch
        num_feat = points_subsets[group_by].dtype.categories.size  # pyright: ignore[reportAttributeAccessIssue]
        columns = points_subsets[group_by].dtype.categories  # pyright: ignore[reportAttributeAccessIssue]
        return gpd.GeoDataFrame(np.zeros((num_tokens or 0, num_feat)), columns=columns).astype(int)

    if pooling in ['cnt']:
        # NOTE: this should also work with varying observed feature_names if the downstream points_encoder can handle it
        tokens = points_subsets.groupby(['token_index', group_by], observed=False).size().unstack().fillna(0).astype(int)
        # TODO: we could support other simple pooling strategies by doing:
        # points_subsets.assing(count=1).pivot_table(index='patch_index', columns=group_by, values='count', aggfunc=pooling)
    else:
        raise NotImplementedError(f'{pooling} is not implemented.')

    if num_tokens:
        if len(tokens) == num_tokens:
            return tokens
        return tokens.reindex(range(num_tokens), fill_value=0)
    return tokens


def to_csr(data: np.ndarray | pd.DataFrame, gene_dict: dict, gene_names: list[str] | None = None):
    """Convert the expression data to a sparse matrix format.
    Args:
        data: The expression data.
        gene_dict: A dictionary mapping gene names to their indices.
    Returns:
        The sparse matrix representation of the expression data.
    """
    from scipy.sparse import csr_matrix

    if isinstance(data, pd.DataFrame):
        valid_genes = [gene for gene in data.columns if gene in gene_dict]  # type: ignore[attr-defined]
        data = data[valid_genes]
        data = data.values  # type: ignore[attr-defined]
    elif gene_names is None:
        raise ValueError('If data is not a DataFrame, gene_names must be provided.')
    else:
        mask = np.isin(gene_names, list(gene_dict.keys()))
        valid_genes = np.array(gene_names)[mask].tolist()
        data = data[:, mask]

    col_idx_to_gene_idx = {i: gene_dict[k] for i, k in enumerate(valid_genes)}

    nonzero_mask = data != 0
    row_indices, cols = np.nonzero(nonzero_mask)
    vals = data[nonzero_mask]
    col_indices = [col_idx_to_gene_idx[col] for col in cols]

    return csr_matrix((vals, (row_indices, col_indices)), shape=(len(data), len(gene_dict)))

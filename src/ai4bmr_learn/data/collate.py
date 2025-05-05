import geopandas as gpd
import pandas as pd
from torch.utils.data._utils.collate import collate, default_collate_fn_map

def collate_dataframe_fn(batch, collate_fn_map=None):
    return batch


custom_collate_fn_map = default_collate_fn_map.copy()
custom_collate_fn_map.update({gpd.GeoDataFrame: collate_dataframe_fn})
custom_collate_fn_map.update({pd.DataFrame: collate_dataframe_fn})


def collate_dataframe(batch, *, collate_fn_map=None):
    """
    Custom collate function that handles GeoPandas GeoDataFrames in PyTorch DataLoaders.

    Args:
        batch: A list of samples fetched from dataset
        collate_fn_map: Optional collate function mapping

    Returns:
        Properly collated batch
    """
    fn_map = collate_fn_map if collate_fn_map is not None else custom_collate_fn_map
    return collate(batch, collate_fn_map=fn_map)
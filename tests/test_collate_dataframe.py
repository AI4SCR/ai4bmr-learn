from ai4bmr_learn.data.collate import collate_dataframe
import pandas as pd
import geopandas as gpd

def test_collate_dataframe():
    df = pd.DataFrame()
    gdf = gpd.GeoDataFrame()
    item = dict(df=df, gdf=gdf)

    batch = collate_dataframe([item] * 2)
    assert len(batch) == 2
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point
import time
from pathlib import Path

# Output directory
base_dir = Path('/work/FAC/FBM/DBC/mrapsoma/prometex/data/benchmark')
base_dir.mkdir(parents=True, exist_ok=True)

# Bounding boxes within 1000x1000 coordinate space
h, w = 100_000, 100_000
bboxs = []
for size in [256, 512]:
    xmin = np.random.uniform(0, w - size)
    ymin = np.random.uniform(0, h - size)
    xmax = xmin + size
    ymax = ymin + size
    bboxs.append((xmin, ymin, xmax, ymax))

# Output paths
test_path_parquet = base_dir / 'with_bbox.parquet'
test_path_gpkg = base_dir / 'with_bbox.gpkg'
test_path_feather = base_dir / 'with_bbox.feather'

# Generate dummy data: 10 million random points
N = 10_000_000
np.random.seed(42)
x = np.random.uniform(0, 1000, N)
y = np.random.uniform(0, 1000, N)
geometries = [Point(x_, y_) for x_, y_ in zip(x, y)]

df = gpd.GeoDataFrame(pd.DataFrame({'id': range(N)}), geometry=geometries, crs="EPSG:4326")

# Export to different formats
df.to_parquet(test_path_parquet, geometry_encoding='geoarrow')
df.to_file(test_path_gpkg, driver='GPKG')
# df.to_feather(test_path_feather)

# Benchmarking function
def benchmark(method_name, func, num_iters=1024):
    start = time.time()
    for _ in range(num_iters):
        result = func()
    end = time.time()
    print(f"{method_name}: {len(result)} points in {end - start:.2f} sec; {num_iters} iterations")

# Run benchmarks
for i, (xmin, ymin, xmax, ymax) in enumerate(bboxs):
    print(f"\n--- Bounding Box {i+1}: ({xmin}, {ymin}, {xmax}, {ymax}) ---")

    # Full load and filter in memory
    benchmark("Full load + in-memory filter", lambda: df[
        df.geometry.x.between(xmin, xmax) & df.geometry.y.between(ymin, ymax)
    ])

    # Parquet + geoarrow (lazy spatial filter)
    benchmark("GeoArrow Parquet + BBOX", lambda: gpd.read_parquet(
        test_path_parquet, bbox=(xmin, ymin, xmax, ymax))
    )

    # GPKG + bbox
    benchmark("GeoPackage (GPKG) + BBOX", lambda: gpd.read_file(
        test_path_gpkg, bbox=(xmin, ymin, xmax, ymax))
    )

    # Feather (entire file in memory + in-memory filter)
    # benchmark("Feather + in-memory filter", lambda: gpd.read_feather(
    #     test_path_feather
    # ).loc[
    #     lambda d: d.geometry.x.between(xmin, xmax) & d.geometry.y.between(ymin, ymax)
    # ])
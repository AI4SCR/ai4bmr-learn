import numpy as np


def find_contours(mask: np.ndarray, fill_holes: bool = True, min_area: int = 500):
    import cv2
    import numpy as np
    from loguru import logger
    import geopandas as gpd
    from shapely.geometry import Polygon, MultiPolygon

    if fill_holes:
        holes, _ = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        for hole in holes:
            cv2.drawContours(mask, [hole], 0, 1, -1)

    if min_area:
        from scipy import ndimage

        label_objects, num_features = ndimage.label(mask)
        sizes = np.bincount(label_objects.ravel())
        mask_sizes = sizes > min_area
        mask_sizes[0] = 0
        mask = mask_sizes[label_objects]
        mask = mask.astype(np.uint8)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        logger.warning("No contours found")
        gdf = gpd.GeoDataFrame(columns=["geometry", "tissue_id"])
        return gdf

    polygons = []
    for contour in contours:
        polygon = np.array(contour.squeeze())
        polygon = np.append(polygon, polygon[0].reshape(1, -1), axis=0)
        polygon = Polygon(polygon)

        if not polygon.is_valid:
            # attempt to fix the polygon by buffering
            polygon = [
                polygon.buffer(i) for i in [0, 0.1, -0.1] if polygon.buffer(i).is_valid
            ]
            polygon = polygon[0]
            if isinstance(polygon, MultiPolygon):
                # extract largest polygon if multiple polygons are found
                polygon = list(polygon.geoms)
                polygon = sorted(polygon, key=lambda x: x.area, reverse=True)[0]

        if polygon.is_valid:
            polygons.append(polygon)

    assert len(polygons) > 0, "No valid polygons"

    gdf = gpd.GeoDataFrame(geometry=polygons)
    gdf["tissue_id"] = range(len(polygons))
    return gdf

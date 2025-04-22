from ai4bmr_learn.utils.images import get_thumbnail, get_thumbnail_size_and_scale
import numpy as np

def visualize_points(points, *, slide=None, image=None, feature_names: list[str] = None,
                     num_points: int=None, max_size=1000,
                     color_by_feature_name: bool=False, legend: bool=False):
    import cv2
    import colorcet as cc

    canvas, scale_factor = get_thumbnail(slide=slide, image=image)

    # Filter by feature names
    if feature_names is not None:
        filter_ = points.feature_name.isin(feature_names)
        points = points[filter_]

    if num_points is not None:
        points = points.sample(num_points)

    # Prepare color mapping from colorcet
    default_color = (0, 0, 255)
    if color_by_feature_name:
        unique_features = sorted(points.feature_name.unique())
        glasbey_colors = cc.glasbey_bw[:len(unique_features)]
        color_map = {
            # feat: tuple(int(255 * c) for c in reversed(rgb))  # RGB -> BGR for OpenCV
            feat: tuple(int(255 * c) for c in rgb)
            for feat, rgb in zip(unique_features, glasbey_colors)
        }
    else:
        unique_features = sorted(points.feature_name.unique())
        color_map = dict()

    # coordinates and scale
    coords = np.vstack([points.geometry.x, points.geometry.y]).T
    coords = (coords * scale_factor).round().astype(int)

    # points
    for (x, y), feat in zip(coords, points.feature_name):
        color = color_map[feat] if color_by_feature_name else default_color
        cv2.circle(canvas, (x, y), radius=2, color=color, thickness=-1)

    # legend
    if legend:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        y_offset = 15
        for idx, feat in enumerate(unique_features):
            color = color_map.get(feat, default_color)
            position = (2, 5 + y_offset * (idx + 1))
            cv2.putText(canvas, str(feat), position, font, font_scale, color, thickness=1, lineType=cv2.LINE_AA)

    return canvas
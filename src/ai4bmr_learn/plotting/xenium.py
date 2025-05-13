from ai4bmr_learn.utils.images import get_thumbnail, get_thumbnail_size_and_scale
import numpy as np

def visualize_points(points, *, slide=None, image=None,
                     include_labels: list[str] = None, labels_key: str = 'feature_name',
                     num_points: int=None, max_size=1000, radius: int = 2, thickness: int = -1,
                     color_by_label: bool=False, color: tuple = (0, 0, 255), color_map: dict, legend: bool=False):
    import cv2
    import colorcet as cc
    from itertools import repeat

    canvas, scale_factor = get_thumbnail(slide=slide, image=image)

    # Filter by labels
    if include_labels is not None:
        filter_ = points[labels_key].isin(include_labels)
        points = points[filter_]

    if num_points is not None:
        num_points = min(num_points, len(points))
        points = points.sample(num_points)

    # Prepare color mapping from colorcet
    default_color = (0, 0, 255)
    if color_by_label:
        from ai4bmr_learn.plotting.utils import get_colorcet_map
        color_map = color_map or get_colorcet_map(points[labels_key], as_int=True)

    # coordinates and scale
    coords = np.vstack([points.geometry.x, points.geometry.y]).T
    coords = (coords * scale_factor).round().astype(int)

    # points
    colors = [color_map[label] for label in points[labels_key]] if color_by_label else [color] * len(coords)
    for (x, y), color in zip(coords, colors):
        cv2.circle(canvas, (x, y), radius=radius, color=color, thickness=thickness)

    # legend
    if legend and color_by_label:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        y_offset = 15
        for idx, label in enumerate(color_map.values()):
            color = color_map.get(label, default_color)
            position = (2, 5 + y_offset * (idx + 1))
            cv2.putText(canvas, str(label), position, font, font_scale, color, thickness=1, lineType=cv2.LINE_AA)

    return canvas
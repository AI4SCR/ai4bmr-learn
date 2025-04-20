from ai4bmr_learn.utils.images import get_thumbnail, get_thumbnail_size_and_scale
import numpy as np

def visualize_transcripts(slide, transcripts, feature_name: str = None, num_transcripts: int = None, max_size=1000):
    import cv2

    canvas = get_thumbnail(slide=slide)
    canvas = np.asarray(canvas).copy()

    # scale_factor
    level = 0
    size = slide.level_dimensions[level]
    _, scale_factor = get_thumbnail_size_and_scale(size=size, max_size=max_size)

    # filter transcripts
    if feature_name is not None:
        filter_ = transcripts.feature_name == feature_name
        transcripts = transcripts[filter_]

    if num_transcripts is not None:
        transcripts = transcripts.sample(num_transcripts)

    # overlay transcripts
    points = (transcripts[['he_x', 'he_y']] * scale_factor).round().astype(int).values
    for x, y in points:
        cv2.circle(canvas, (x, y), radius=2, color=(0, 0, 255), thickness=-1)

    return canvas

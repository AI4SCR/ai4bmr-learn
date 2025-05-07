from openslide import OpenSlide
import json
from dataclasses import asdict
from pathlib import Path
from tqdm import tqdm
import einops
import openslide
import torch
from ai4bmr_learn.utils.device import get_device
from ai4bmr_learn.datasets.Patches import Patches
from ai4bmr_learn.plotting.contours import visualize_contours
from ai4bmr_learn.plotting.patches import visualize_coords
from ai4bmr_learn.preprocessing.padding import get_pad_for_kernel
from ai4bmr_learn.utils.contours import find_contours
from ai4bmr_learn.utils.images import (
    SlideCoordinate,
    get_coordinates_dict,
    get_slide_patcher_params,
    get_thumbnail,
)

from loguru import logger
from shapely.affinity import scale
from skimage.io import imsave
from torchvision.transforms import v2


def get_mpp_and_resolution(slide: OpenSlide):
    res_unit = slide.properties['tiff.ResolutionUnit']
    x_res = float(slide.properties['tiff.XResolution'])
    y_res = float(slide.properties['tiff.YResolution'])
    assert x_res == y_res
    res = int(x_res)

    match res_unit:
        case 'centimeter':
            mpp = 1e4 / res
        case _:
            raise ValueError(f"Unknown resolution unit: {res_unit}")

    return mpp, res


def get_mpp(slide: OpenSlide):
    mpp_x = float(slide.properties['openslide.mpp-x'])
    mpp_y = float(slide.properties['openslide.mpp-y'])
    assert mpp_x == mpp_y

    return mpp_x


def get_seg_model(model_name: str = 'hest', device: str = 'cuda') -> torch.nn.Module:
    """get a segmentation model from the trident library
    args:
        model_name (str): name of the model to load
        device (str): device to load the model on
    """
    try:
        from trident.segmentation_models import (  # pants: no-infer-dep  # pyright: ignore[reportMissingImports]
            segmentation_model_factory,
        )  # pants: no-infer-dep  # pyright: ignore[reportMissingImports]
    except ImportError as e:
        logger.warning('Trident library not found. Please install it to use the segmentation model.')
        raise e

    # MODEL
    return segmentation_model_factory(model_name).eval().to(device=device)  # type: ignore[reportGeneralTypeIssues]


def segment_slide(slide: openslide.OpenSlide,
                  target_mpp: float = 4,
                  min_area: float = 500,
                  seg_model: torch.nn.Module | None = None,
                  save_contours_path: Path | None = None,
                  save_coords_path: Path | None = None,
                  batch_size: int = 32,
                  num_workers: int = 8,
                  **kwargs
                  ):
    """
    Segments histological images in a given sample directory using a pretrained model.
    Saves the resulting contours, mask overlays, and patch coordinates.
    Args:
        sample_dir (Path): Directory containing the sample with 'region.tif'.
    """

    # MODEL
    device = get_device(logger=logger)
    if seg_model is None:
        seg_model = get_seg_model('hest', device=device)
    size = seg_model.input_size

    transform = v2.Compose(
        [
            v2.Resize(size=size, interpolation=v2.InterpolationMode.BILINEAR),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    # %% COORDS
    patch_size = patch_stride = size
    params = get_slide_patcher_params(
        slide=slide,
        patch_size=patch_size,
        patch_stride=patch_stride,
        target_mpp=target_mpp,
    )

    coords = get_coordinates_dict(**params, include_out_of_bounds=True)
    coords = [SlideCoordinate(**i) for i in coords]

    save_coords_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_coords_path, 'w') as f:
        coords_dict = [asdict(i) for i in coords]
        json.dump(coords_dict, f)

    # %% DATASET
    ds = Patches(coords=coords, transform=transform)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                     pin_memory=num_workers > 0)

    # %% SEGMENT
    mask = []
    with torch.no_grad():
        for batch in tqdm(dl, desc='Segmenting'):
            images = batch['image']
            mask.append(seg_model(images.to(device=device)).cpu())

    mask = torch.cat(mask, dim=0)

    # %% MASK
    height, width, kernel_size, stride = (params['height'], params['width'], params['kernel_size'], params['stride'])
    ph, pw = get_pad_for_kernel(height=height, width=width, kernel_size=kernel_size, stride=stride)

    height = height + ph
    width = width + pw

    h = height // kernel_size
    assert height % kernel_size == 0

    w = width // kernel_size
    assert width % kernel_size == 0

    # note: patches -> mask; requires the image to be patched in row-major order
    mask = einops.rearrange(mask, '(h w) ph pw -> (h ph) (w pw)', h=h, w=w, ph=size, pw=size)

    # CONTOURS
    gdf = find_contours(mask=mask.numpy(), min_area=min_area)
    scale_factor = params['scale_factor']
    gdf['geometry'] = gdf['geometry'].map(
        lambda geom: scale(geom, xfact=scale_factor, yfact=scale_factor, origin=(0, 0))
    )
    save_contours_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_parquet(save_contours_path)

    # VISUALIZE
    thumbnail, _ = get_thumbnail(slide=slide)
    canvas = visualize_coords(coords=coords, slide=slide, image=thumbnail)
    save_coords_overlay = save_coords_path.parent / f'{save_coords_path.stem}.png'
    imsave(str(save_coords_overlay), canvas)

    canvas = visualize_contours(contours=gdf, slide=slide, image=thumbnail, **kwargs)
    save_contours_overlay = save_contours_path.parent / f'{save_contours_path.stem}.png'
    imsave(str(save_contours_overlay), canvas)

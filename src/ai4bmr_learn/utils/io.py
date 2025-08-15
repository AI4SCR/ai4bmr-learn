from pathlib import Path
from PIL.Image import Image
import numcodecs
import numpy as np

# Common compressor: Blosc with Zstd + shuffle
BLOSC = numcodecs.Blosc(cname="zstd", clevel=3, shuffle=numcodecs.Blosc.SHUFFLE)

def imread(img_path: Path, engine: str | None = None):
    engine = engine or img_path.suffix
    match engine:
        case '.tiff':
            import tifffile
            data = tifffile.imread(img_path)
        case '.tif':
            import tifffile
            data = tifffile.imread(img_path)
        case '.zip':
            import zarr
            store = zarr.ZipStore(str(img_path), mode="r")
            data = zarr.open_array(store=store)
        case '.zarr':
            import zarr
            data = zarr.open_array(img_path)
        case _:
            raise ValueError(f'Unknown image engine {engine} for {img_path}')
    return np.array(data)

def save_image(
        img: np.ndarray,
        save_path: Path,
        chunks: tuple[int, int, int] | None = None,
        compressor=BLOSC,
        engine: str | None = None
):
    """
    Save a CHW image or mask array to a Zarr store.
    Mask arrays can be int16 just by passing arr.dtype == np.int16.
    """
    chunks = chunks or (img.shape[0], 256, 256)
    engine = engine or save_path.suffix
    match engine:
        case '.zarr':
            import zarr
            zarr.save_array(store=str(save_path), arr=img, compressor=compressor)
        case '.zip':
            import zarr
            store = zarr.ZipStore(str(save_path), mode="r")
            zarr.open_array(store=store, arr=img, compressor=compressor)
        case '.tiff':
            import tifffile
            tifffile.imwrite(save_path, img, tile=chunks, compression=compressor)
        case '.tif':
            import tifffile
            tifffile.imwrite(save_path, img, tile=chunks, compression=compressor)


imsave = save_image

def read_region(
        img_path: Path,
        y: int,
        x: int,
        height: int,
        width: int,
        channel: int | slice = slice(None),
        level: int = 0,
        engine: str | None = None
) -> np.ndarray | Image:
    """
    Read a (height × width) patch (or multi-channel patch) from a saved Zarr store.
    """
    engine = engine or img_path.suffix
    match engine:
        case '.zarr':
            import zarr
            z = zarr.open_array(str(img_path), mode='r')
            return z[channel, y: y + height, x: x + width]
        case '.zip':
            import zarr
            store = zarr.ZipStore(str(img_path), mode="r")
            return zarr.open_array(store=store)
        case '.tiff':
            import tifffile
            return tifffile.imread(files=img_path, mode="r")
        case '.tif':
            import tifffile
            return tifffile.imread(files=img_path, mode="r")
        case '.openslide':
            import openslide
            slide = openslide.open_slide(str(img_path))
            img = slide.read_region(location=(x, y), level=level, size=(width, height))
            return img


def save_mask(mask: np.ndarray, save_path: Path):
    import tifffile
    assert mask.dtype in ["uint32", "uint16", 'int64', 'int32']

    if mask.max() < 65536:
        mask = mask.astype("uint16")
    else:
        mask = mask.astype("uint32")

    tifffile.imwrite(save_path, mask, compression="deflate")

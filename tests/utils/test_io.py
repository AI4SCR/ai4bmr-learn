import numpy as np
import pytest
from pathlib import Path
from ai4bmr_learn.utils.io import save_image, imread, read_region


@pytest.fixture
def dummy_image():
    return np.random.randint(0, 256, size=(3, 128, 128), dtype=np.uint8)


@pytest.mark.parametrize("suffix", [".zarr", ".zip", ".tiff"])
def test_save_and_read_image(dummy_image, tmp_path, suffix):
    save_path = tmp_path / f"test_image{suffix}"

    # Save the image
    save_image(dummy_image, save_path)

    # Read the whole image back
    read_img = imread(save_path)

    # TIFF will be read as RGBA by default with openslide, so we compare only RGB
    if suffix in [".tiff", ".tif"]:
        # And tifffile reads it back as (128, 128, 3) so we transpose
        read_img = read_img.transpose(2, 0, 1)
    
    np.testing.assert_array_equal(dummy_image, read_img)

    # Read a region
    y, x, height, width = 10, 20, 30, 40
    region = read_region(save_path, y=y, x=x, height=height, width=width)

    if suffix in [".tiff", ".tif"]:
        # openslide reads region as PIL image, need to convert and transpose
        region = np.array(region).transpose(2, 0, 1)

    expected_region = dummy_image[:, y:y+height, x:x+width]
    np.testing.assert_array_equal(expected_region, region)

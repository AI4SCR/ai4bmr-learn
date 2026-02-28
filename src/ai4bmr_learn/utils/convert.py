from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from loguru import logger


def to_pyramidal(
    img_path: Path,
    save_path: Path,
    tile: int = 512,
    compression: str = "deflate",
    bigtiff: bool = True,
    **kwargs,
) -> Path:
    source = Path(img_path).expanduser().resolve()
    target = Path(save_path).expanduser().resolve()
    assert source.exists(), f"Input image does not exist: {source}"
    assert tile > 0, "tile must be > 0"

    target.parent.mkdir(parents=True, exist_ok=True)

    import pyvips

    image = pyvips.Image.new_from_file(str(source), access="sequential")
    image.tiffsave(
        str(target),
        tile=True,
        tile_width=tile,
        tile_height=tile,
        pyramid=True,
        compression=compression,
        bigtiff=bigtiff,
        **kwargs,
    )
    logger.info(f"Saved pyramidal TIFF to: {target}")
    return target


def to_ome_tiff(
    img_path: Path,
    save_path: Path,
    compression: str = "JPEG",
    quality: int = 95,
    bigtiff: bool = True,
) -> int:
    source = Path(img_path).expanduser().resolve()
    target = Path(save_path).expanduser().resolve()
    assert source.exists(), f"Input image does not exist: {source}"
    assert 0 < quality <= 100, "quality must be in (0, 100]"
    assert shutil.which("bfconvert"), "Missing `bfconvert` in PATH."

    target.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "bfconvert",
        "-noflat",
        *([] if not bigtiff else ["-bigtiff"]),
        "-compression",
        compression,
        "-quality",
        str(quality),
        str(source),
        str(target),
    ]
    result = subprocess.run(cmd, check=True)
    logger.info(f"Saved OME TIFF to: {target}")
    return result.returncode

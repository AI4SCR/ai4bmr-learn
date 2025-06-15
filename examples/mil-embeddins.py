from pathlib import Path

from loguru import logger

class BEAT:

    def __init__(self, base_dir: Path | None = None):
        self.base_dir = base_dir or Path('/work/FAC/FBM/DBC/mrapsoma/prometex/data/beat/')
        self.raw_dir = self.base_dir / '01_raw'
        self.processed_dir = self.base_dir / '02_processed'
        self.tools_dir = self.base_dir / '03_tools'
        self.log_dir = self.base_dir / '04_logs'

        self.images_dir = self.processed_dir / 'images'
        self.images_dir.mkdir(parents=True, exist_ok=True)

    def prepare_tools(self):
        bfconvert = self.tools_dir / "bftools" /"bfconvert"

        if bfconvert.exists():
            logger.info(f"bfconvert found at {bfconvert}. Skipping installation.")
            return

        import textwrap
        import subprocess
        logger.info("⚙️Installing Bio-Formats tools.")
        self.tools_dir.mkdir(parents=True, exist_ok=True)

        install_cmd = f"""
        wget https://downloads.openmicroscopy.org/bio-formats/8.2.0/artifacts/bftools.zip -O {self.tools_dir}/bftools.zip
        unzip {self.tools_dir}/bftools.zip
        {self.tools_dir / "bftools" /"bfconvert"} -version || exit
        """

        install_cmd = textwrap.dedent(install_cmd).strip()
        subprocess.run(install_cmd, shell=True, check=True)


    def prepare_data(self, force: bool = False):
        import textwrap
        import subprocess
        from ai4bmr_core.utils.tidy import tidy_name

        self.prepare_tools()
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        bfconvert = self.tools_dir / "bftools" / "bfconvert"

        wsi_paths = list(self.base_dir.rglob('*.ndpi')) + list(self.base_dir.rglob('*.czi'))
        for wsi_path in wsi_paths:
            save_name = tidy_name(wsi_path.stem)
            save_path = self.images_dir / f'{save_name}.ome.tiff'

            if save_path.exists() and not force:
                logger.info(f"File {save_path} already exists. Skipping conversion.")

            job_name = f"wsi2tiff-{save_name}"
            logger.info(f"Converting {wsi_path} to {save_path} using Bio-Formats (job_name={job_name})")

            sbatch_command = f"""
            sbatch \\
            --job-name={job_name} \\
            --output=/users/amarti51/logs/{job_name}-%A-%a.log \\
            --error=/users/amarti51/logs/{job_name}-%A-%a.err \\
            --time=08:00:00 \\
            --mem=128G \\
            --cpus-per-task=12 \\
            <<'EOF'
            #!/bin/bash

            INPUT="{wsi_path}"
            OUTPUT="{save_path}"
            BFCONVERT="{bfconvert}"

            /usr/bin/time -v "$BFCONVERT" -nogroup -bigtiff -compression LZW -tilex 512 -tiley 512 "$INPUT" "$OUTPUT"
            EOF
            """
            sbatch_command = textwrap.dedent(sbatch_command).strip()

            subprocess.run(sbatch_command, shell=True, check=True)

    def setup(self):
        pass

dm = BEAT()
dm.prepare_data()

# %%
# import openslide
# from ai4bmr_learn.utils.slides import segment_slide, get_seg_model
# base_dir = Path('/work/FAC/FBM/DBC/mrapsoma/prometex/data/beat/')
# raw_dir = base_dir / 'raw'
# save_coords_path = Path('/users/amarti51/prometex/data/H_E_Images/coords.parquet')
# save_contours_path = Path('/users/amarti51/prometex/data/H_E_Images/contours.parquet')
# segment_slide(slide,
#               seg_model=get_seg_model(model_name='grandqc'),  # hest, grandqc, grandqc_artifact
#               target_mpp=4.0,
#               save_contours_path=save_contours_path, save_coords_path=save_coords_path)
#
# import pandas as pd
# pd.Series([i.name for i in Path('/users/amarti51/prometex/data/beat/01_raw/H_E_Images').rglob('*.czi')]).value_counts().max()

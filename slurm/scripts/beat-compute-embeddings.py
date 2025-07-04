# %%
from ai4bmr_datasets import BEAT
dm = self = BEAT()
dm.create_patch_embeddings(batch_size=128, num_workers=12)

from pathlib import Path
from loguru import logger
model_name: str = 'uni_v1'
i = 0
wsi_path = Path('/work/PRTNR/CHUV/DIR/rgottar1/spatial/data/beat/02_processed/datasets/beat/1G4X.06.0/wsi.tiff')

# dm.segment(model_name='hest', target_mpp=4)
# # dm.segment(model_name='hest', target_mpp=4, source_mpp=0.442)
# dm.segment(model_name='grandqc', target_mpp=4)
# # dm.segment(model_name='grandqc', target_mpp=4, source_mpp=0.442)
from ai4bmr_datasets import BEAT
dm = self = BEAT()

# dm.prepare_clinical()
# dm.prepare_wsi()
dm.segment(model_name='hest', target_mpp=4)
dm.segment(model_name='hest', target_mpp=4, source_mpp=0.25)
dm.segment(model_name='grandqc', target_mpp=4)
dm.segment(model_name='grandqc', target_mpp=4, source_mpp=0.25)

# %%
import openslide
slide = openslide.OpenSlide('/work/PRTNR/CHUV/DIR/rgottar1/spatial/data/beat/02_processed/datasets/beat/1IS2.0G.0/wsi.tiff')
slide.level_dimensions

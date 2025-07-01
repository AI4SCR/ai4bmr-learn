# %%
from pathlib import Path
import pandas as pd
import openslide
from ai4bmr_learn.utils.slides import get_mpp
records = []
for wsi_path in Path('/work/PRTNR/CHUV/DIR/rgottar1/spatial/data/beat/02_processed/datasets/beat').rglob('*wsi.tiff'):
    slide = openslide.OpenSlide(wsi_path)
    try:
        mpp = get_mpp(slide)
    except AssertionError as e:
        mpp = 'AssertionError'

    records.append({'wsi_path': wsi_path, 'mpp': mpp})
    print(f'{wsi_path.parent.name}: {mpp}')

mpps = pd.DataFrame.from_records(records)
mpps['sample_id'] = mpps.wsi_path.map(lambda x: x.parent.name)

clinical = pd.read_parquet('/work/PRTNR/CHUV/DIR/rgottar1/spatial/data/beat/02_processed/metadata/clinical.parquet')
clinical = clinical.rename(columns={'wsi_path': 'wsi_path_raw'}).reset_index()

mpps = mpps.merge(clinical, on='sample_id')
mpps = mpps[['wsi_path_raw', 'wsi_path', 'n', 'sample_barcode', 'sample_id', 'mpp']]

mpps.to_csv('/work/PRTNR/CHUV/DIR/rgottar1/spatial/data/beat/02_processed/metadata/mpps.csv', index=False)

# %%
sample_ids_to_investigate = mpps[~mpps.mpp.astype(str).str.startswith('0.')].sample_id.to_list()

# %%
from ai4bmr_datasets import BEAT
dm = self = BEAT()

dm.segment(model_name='hest', target_mpp=4)
# dm.segment(model_name='hest', target_mpp=4, source_mpp=0.442)
dm.segment(model_name='grandqc', target_mpp=4)
# dm.segment(model_name='grandqc', target_mpp=4, source_mpp=0.442)
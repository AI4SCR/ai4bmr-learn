# %%
from ai4bmr_datasets import BEAT
dm = self = BEAT()
dm.create_patch_embeddings(batch_size=128, num_workers=12)

# dm.segment(model_name='hest', target_mpp=4)
# # dm.segment(model_name='hest', target_mpp=4, source_mpp=0.442)
# dm.segment(model_name='grandqc', target_mpp=4)
# # dm.segment(model_name='grandqc', target_mpp=4, source_mpp=0.442)
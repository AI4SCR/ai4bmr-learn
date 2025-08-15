from torch.utils.data import DataLoader
from ai4bmr_learn.models.encoder.geneformer import Geneformer
from ai4bmr_learn.collators.geneformer import GeneformerCollate
from ai4bmr_learn.datasets.coordinates import Coordinates
from pathlib import Path
from ai4bmr_learn.utils.device import batch_to_device

ds = Coordinates(
    coords_path=Path('/work/PRTNR/CHUV/DIR/rgottar1/spatial/data/fmx/data/splits/hest1k-tts=4-fvs=0-min_transcripts_per_patch=200/test-0.json'))
ds.setup()
ds[0]

collate_fn = GeneformerCollate(kernel_size=256, stride=256)
batch = next(iter(DataLoader(ds, batch_size=2, collate_fn=collate_fn)))
batch = batch_to_device(batch, device='cuda')

model = Geneformer(model_name='gf-12L-30M-i2048')
model.to('cuda')
input_ids, attention_mask = batch['expression']['input_ids'], batch['expression']['attention_mask']
out = model.forward(input_ids=input_ids, attention_mask=attention_mask)

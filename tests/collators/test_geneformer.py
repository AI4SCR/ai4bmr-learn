def test_geneformer_collate():
    from ai4bmr_learn.collators.geneformer import GeneFormerCollate
    import geopandas as gpd
    import numpy as np

    ensembl_ids = [
        "ENSG00000139618",
        "ENSG00000141510",
        "ENSG00000157764",
        "ENSG00000155657",
        "ENSG00000134982"
    ]

    num_obs = 100
    x = np.random.randint(0, 224, size=num_obs)
    y = np.random.randint(0, 224, size=num_obs)

    points = gpd.GeoDataFrame({'geometry': gpd.points_from_xy(x, y)})
    points['ensembl_id'] = np.random.choice(ensembl_ids, size=num_obs, replace=True)
    points['transcript_id'] = np.arange(num_obs)

    collator = GeneFormerCollate(kernel_size=16, stride=16)

    item1 = {'points': points, 'patch_size': 224}
    item2 = {'points': points.copy(), 'patch_size': 224}

    batch = collator([item1, item2])

    assert 'expression' not in batch
    assert 'attention_masks' in batch['expression']
    assert 'input_ids' in batch['expression']
    assert batch['expression']['input_ids'].shape[:2] == (2, (224 // 16) ** 2)  # pyright: ignore

    from ai4bmr_learn.models.encoder.geneformer import GeneformerHelical
    import torch
    
    model = GeneformerHelical(device='cpu')
    out = model.forward(**batch['expression'])
    out = torch.nanmean(out, dim=(1, 2))

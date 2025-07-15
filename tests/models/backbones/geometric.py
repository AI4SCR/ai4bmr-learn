def gcn():
    import torch_geometric
    from ai4bmr_learn.models.backbones.geometric import GCN
    from torch.utils.data import DataLoader
    from ai4bmr_learn.collators.graph import Graph
    from pathlib import Path

    import torch
    import tifffile
    mask1 = tifffile.imread('/users/amarti51/prometex/data/dinov1/datasets/Cords2024/masks/175_A_1.tiff')
    mask2 = tifffile.imread('/users/amarti51/prometex/data/dinov1/datasets/Cords2024/masks/175_A_100.tiff')
    item1 = {'image': torch.randn(43, *mask1.shape), 'mask': torch.tensor(mask1)}
    item2 = {'image': torch.randn(43, *mask2.shape), 'mask': torch.tensor(mask2)}

    stats_path = Path('/users/amarti51/prometex/data/dinov1/datasets/Cords2024/images/default/stats.json')
    collator = Graph(radius=32, include_self=True, stats_path=stats_path)

    data_list = [collator.process_item(item1)['graph'], collator.process_item(item2)['graph']]
    loader = torch_geometric.loader.DataLoader(data_list, batch_size=2)
    batch = next(iter(loader))
    batch.to('cuda')

    model = GCN(num_node_features=48, embed_dim=512)
    model.to('cuda')

    out = model(batch)
    out.shape

    item1 = {'image': torch.randn(43, *mask1.shape), 'mask': torch.tensor(mask1)}
    item2 = {'image': torch.randn(43, *mask2.shape), 'mask': torch.tensor(mask2)}
    dl = DataLoader([item1, item2], batch_size=2, collate_fn=collator)
    batch = next(iter(dl))
    data = batch['graph']
    data.to('cuda')

    out = model(data)
    out.shape

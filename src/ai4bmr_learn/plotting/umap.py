
def run_umap(data, n_neighbors= 15, min_dist=0.3, metric='euclidean', engine: str = 'umap-learn', **kwargs):
    import pandas as pd

    data = data.values if isinstance(data, pd.DataFrame) else data

    match engine:
        case 'cuml':
            from cuml.manifold.umap import UMAP
            reducer = UMAP(n_neighbors=n_neighbors, min_dist=min_dist, metric=metric, **kwargs)
            reducer.fit(data)
            return reducer

        case 'umap-learn':
            from umap import UMAP
            reducer = UMAP(n_neighbors=n_neighbors, min_dist=min_dist, metric=metric, **kwargs)
            reducer.fit(data)
            return reducer

        case 'rapids-sc':
            import rapids_singlecell as rsc
            from anndata import AnnData

            ad = AnnData(X=data, obs=None)
            rsc.get.anndata_to_GPU(ad)
            rsc.pp.neighbors(ad, n_neighbors=n_neighbors, metric=metric, use_rep='X', **kwargs)
            rsc.tl.umap(ad, min_dist=min_dist)
            return ad

        case _:
            raise ValueError(f"Unknown engine: {engine}")

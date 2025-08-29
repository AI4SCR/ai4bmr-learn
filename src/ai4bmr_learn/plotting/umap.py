from matplotlib import pyplot as plt
import numpy as np

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

def plot_umap(data, *,
              n_neighbors: int = 15, min_dist: float = 0.3, metric: str = 'euclidean', engine: str = 'umap-learn', umap_kwargs: dict | None = None, return_reducer: bool = False,
              ax: plt.Axes | None = None, labels: np.ndarray | list[str, int] | None = None, values: np.ndarray | list[int, float] | None = None, num_samples: int | None = None, show_legend: bool = True, **kwargs):

        import pandas as pd
        import numpy as np
        import umap.plot

        umap_kwargs = umap_kwargs or {}

        data = data.values if isinstance(data, pd.DataFrame) else data
        labels = labels.values if isinstance(labels, pd.Series) else labels
        values = values.values if isinstance(values, pd.Series) else values

        if num_samples and len(data) > num_samples:
            indices = np.random.choice(len(data), num_samples, replace=False)
            data = data[indices]
            labels = labels[indices] if labels is not None else None
            values = values[indices] if values is not None else None

        if ax is None:
            _, ax = plt.subplots()

        reducer = run_umap(data, n_neighbors=n_neighbors, min_dist=min_dist, metric=metric, engine=engine, **umap_kwargs)

        match engine:
            case 'rapids-sc':
                import scanpy as sc

                if labels is not None:
                    reducer.obs['labels'] = labels
                    color = 'labels'
                elif values is not None:
                    reducer.obs['values'] = values
                    color = 'values'
                else:
                    color = None

                legend_loc = 'right margin' if show_legend else None
                sc.pl.umap(reducer, ax=ax, color=color, show=False, legend_loc=legend_loc, color_map='inferno', **kwargs)
                ax.set_facecolor('black')

            case _:
                umap.plot.points(reducer,
                                 labels=labels, values=values,
                                 background='black', cmap='inferno',
                                 ax=ax, show_legend=show_legend, **kwargs)

        return (ax, reducer) if return_reducer else ax

def csr_to_precomputed_knn(csr, k):
    """
    Convert a CSR matrix to a tuple of indices and distances for UMAP precomputed knn.
    """
    indices = csr.indices.reshape(-1, k)
    distances = csr.data.reshape(-1, k)
    return indices, distances
# %%
import igraph as ig
import leidenalg as la
import numpy as np
import pandas as pd
import scipy.sparse
import umap
from loguru import logger
from ai4bmr_learn.plotting.umap import plot_umap

def kneighbors_graph(data, n_neighbors: int = 15, metric: str = 'euclidean', mode: str = 'connectivity', engine: str = 'sklearn', algorithm: str = 'auto', **kwargs):
    """
    Compute a graph from data using different methods and engines.

    Args:
        data (array-like): Input data matrix.
        engine (str): Backend engine to use ('rapids', 'rapids-sc', or 'sklearn').
        kwargs: Additional keyword arguments (e.g., n_neighbors).

    Returns:
        csr_matrix: Sparse adjacency matrix.
    """

    import pandas as pd

    data = data.values if isinstance(data, pd.DataFrame) else data

    if engine == 'sklearn':
        from sklearn.neighbors import NearestNeighbors

        kwargs = {'n_neighbors': n_neighbors, 'metric': metric, 'algorithm': algorithm, **kwargs}
        nn = NearestNeighbors(**kwargs)
        nn.fit(data)
        csr = nn.kneighbors_graph(data, mode=mode)

    elif engine == 'cuml':
        from cuml.neighbors import NearestNeighbors
        import cupy as cp

        data = cp.asarray(data)

        kwargs = {'n_neighbors': n_neighbors, 'metric': metric, 'algorithm': algorithm, **kwargs}
        nn = NearestNeighbors(**kwargs)
        nn.fit(data)
        csr = nn.kneighbors_graph(data, mode=mode)
        csr = csr.get()

    elif engine == 'pydecent':
        raise NotImplementedError()

    elif engine == 'rapids-sc' or engine == 'scanpy':
        from anndata import AnnData
        ad = AnnData(X=data, obs=None)

        if engine == 'rapids-sc':
            import rapids_singlecell as sc
            sc.get.anndata_to_GPU(ad)
        else:
            import scanpy as sc

        algorithm = 'brute' if algorithm in ['brute', 'auto'] else algorithm
        sc.pp.neighbors(ad, n_neighbors=n_neighbors, metric=metric, use_rep='X', **kwargs)

        if mode == 'connectivity':
            csr = ad.obsp["connectivities"]
        elif mode == 'distance':
            csr = ad.obsp["distances"]

    else:
        raise ValueError(f"Unknown engine '{engine}'. Choose from 'sklearn', 'cuml', 'scanpy', 'rapids-sc' or 'pydecent'.")

    return csr


def get_membership_from_data(
        data: pd.DataFrame | np.ndarray,
        resolution: float = 1,
        n_neighbors: int = 15,
        method: str = None,
        graph_engine: str = "cuml",
        graph_kwargs: dict = None,
        membership_engine: str = "igraph",
        membership_kwargs: dict = None,
):

    data = data.values if isinstance(data, pd.DataFrame) else data

    if graph_kwargs is None:
        graph_kwargs = dict()

    if membership_kwargs is None:
        membership_kwargs = dict()

    if method in ['scanpy', 'rapids-sc']:
        from anndata import AnnData

        logger.info(f'Using {method} for membership computation.')

        ad = AnnData(X=data, obs=None)

        if method == 'rapids-sc':
            import rapids_singlecell as sc
            sc.get.anndata_to_GPU(ad)
        elif method == 'scanpy':
            import scanpy as sc
        else:
            raise ValueError(f"Unknown method '{method}'. Choose from 'scanpy' or 'rapids-sc'.")

        sc.pp.neighbors(ad, n_neighbors=n_neighbors, use_rep="X", **graph_kwargs)
        sc.tl.leiden(ad, resolution=resolution, **membership_kwargs)
        membership = ad.obs["leiden"].astype(str).values

    else:
        csr = kneighbors_graph(data, n_neighbors=n_neighbors, engine=graph_engine, **graph_kwargs)
        graph = csr_to_ig(csr, directed=False, weighted=False)
        membership = get_membership(graph, resolution=resolution, engine=membership_engine, **membership_kwargs)

    return membership


def csr_to_ig(csr: scipy.sparse.csr_matrix, directed=False, weighted=False):

    csr = csr.tocoo()
    edges = zip(csr.row, csr.col)

    if weighted:
        logger.warning(f'using weighted graph with weights 1 / distance. This is experimental.')
        weights = csr.data ** -1
    else:
        weights = np.ones_like(csr.data)

    g = ig.Graph(edges=edges, directed=directed)
    g.es["weight"] = weights

    return g


def get_membership(graph, resolution: float = 1, engine: str = "igraph"):
    """
    Get the membership of the graph using the Leiden algorithm.

    Args:
        G (igraph.Graph): The input graph.
        resolution (float): Resolution parameter for the Leiden algorithm.

    Returns:
        list: Membership of each vertex in the graph.
    """

    match engine:

        case "igraph":
            return get_igraph_membership(graph, resolution=resolution)
        case "leiden":
            return get_leiden_membership(graph, resolution=resolution)
        case _:
            raise ValueError(f"Unknown engine '{engine}'. Choose from 'igraph' or 'leiden'.")


def get_leiden_membership(G, resolution: float = 1):
    partition = la.find_partition(
        G,
        la.RBConfigurationVertexPartition,
        weights=G.es["weight"],
        resolution_parameter=resolution,
    )
    return partition.membership


def get_igraph_membership(G, resolution: float = 1):
    return G.community_leiden(objective_function="modularity",
                              resolution=resolution,
                              weights="weight").membership


def get_kmeans_membership(data, n_clusters: int = 20):
    from sklearn.cluster import KMeans

    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(data)
    return kmeans.labels_

# %%
import igraph as ig
import leidenalg as la
import numpy as np
import pandas as pd
import scipy.sparse
import umap
from loguru import logger

def compute_knn_graph(data, n_neighbors: int = 15, metric: str = 'euclidean', mode: str = 'connectivity', engine: str = 'sklearn', **kwargs):
    """
    Compute a graph from data using different methods and engines.

    Args:
        data (array-like): Input data matrix.
        engine (str): Backend engine to use ('rapids', 'pydecent', or 'sklearn').
        kwargs: Additional keyword arguments (e.g., n_neighbors).

    Returns:
        csr_matrix: Sparse adjacency matrix.
    """

    import pandas as pd

    data = data.values if isinstance(data, pd.DataFrame) else data

    if engine == 'sklearn':
        from sklearn.neighbors import kneighbors_graph

        kwargs = {'n_neighbors': n_neighbors, 'metric': metric, 'mode': mode, **kwargs}
        csr = kneighbors_graph(data, **kwargs)

    elif engine == 'cuml':
        from cuml.neighbors import NearestNeighbors

        kwargs = {'n_neighbors': n_neighbors, 'metric': metric, **kwargs}
        nn = NearestNeighbors(**kwargs)
        nn.fit(data)
        csr = nn.kneighbors_graph(data, mode=mode)
        csr = csr.get()  # to cpu

    elif engine == 'rapids-sc':
        import rapids_singlecell as rsc
        from anndata import AnnData

        ad = AnnData(X=data, obs=None)
        rsc.get.anndata_to_GPU(ad)
        rsc.pp.neighbors(ad, n_neighbors=n_neighbors, metric=metric, use_rep='X', **kwargs)
        if mode == 'connectivity':
            csr = ad.obsp["connectivities"]
        elif mode == 'distance':
            csr = ad.obsp["distances"]

    else:
        raise ValueError(f"Unknown engine '{engine}'. Choose from 'sklearn', 'rapids', or 'pydecent'.")

    return csr



def cluster(
        data: pd.DataFrame,
        resolution: float = 1,
        n_neighbors: int = 15,
        graph_method: str = "rapids",
        graph_kwargs: dict = None,
        cluster_method: str = "igraph",
        clutering_kwargs: dict = None,
):
    logger.info(f"cluster data with {len(data)} observations")
    logger.info(f"compute graph with method `{graph_method}`")

    if graph_kwargs is None:
        graph_kwargs = dict()

    if clutering_kwargs is None:
        clutering_kwargs = dict()

    # %%
    if graph_method == "rapids":
        kwargs = dict(
            n_neighbors=n_neighbors,
            min_dist=0.1,
                      metric="euclidean",
                      n_components=2)
        kwargs.update(graph_kwargs)

        mapper = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
            n_components=n_components,
            n_jobs=-1,
        )
        mapper.fit(data)
        csr = mapper.graph_
    elif graph_method == "knn":
        from sklearn.neighbors import kneighbors_graph
        metric = "euclidean"
        csr = kneighbors_graph(data, n_neighbors=n_neighbors, mode="connectivity", metric=metric, include_self=False)
        # assert False, 'we need to convert distance to weights'
    elif graph_method == "rapids-sc":
        pass
    elif graph_method == "scanpy":
        from anndata import AnnData
        import scanpy as sc
        from numba import cuda

        sc.pp.neighbors(data)
        if cuda.is_available():
            logger.info("GPU available")
            import rapids_singlecell as sc
        else:
            import scanpy as sc

        # %% create AnnData
        ad = AnnData(X=data.values, obs=None)

        if cuda.is_available():
            sc.get.anndata_to_GPU(ad)
        sc.pp.neighbors(ad, n_neighbors=n_neighbors, use_rep="X")
        csr = ad.obsp["connectivities"]

    G = csr_to_ig(csr)

    if cluster_method == "leiden":
        membership = get_leiden_membership(G, resolution=resolution)
    elif cluster_method == "kmeans":
        n_clusters = 20
        membership = get_kmeans_membership(data, n_clusters=n_clusters)
    elif cluster_method == "leiden-scanpy":
        assert graph_method == "scanpy"
        sc.tl.leiden(ad, resolution=resolution)
        # sc.tl.leiden(ad, resolution=resolution, flavor="igraph", n_iterations=2)
        membership = ad.obs["leiden"].astype(str).values

    membership = pd.Categorical([str(i) for i in membership])

    num_cluster = np.unique(membership).size
    metadata = pd.Series(membership, index=data.index, name="membership")
    result = dict(num_obs=len(data), num_cluster=num_cluster)

    logger.info("completed")
    return metadata


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

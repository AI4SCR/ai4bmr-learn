# %%
import igraph as ig
import leidenalg as la
import numpy as np
import pandas as pd
import scipy.sparse
import umap
from numba import cuda
from loguru import logger


def cluster(
    data: pd.DataFrame,
    metadata: pd.DataFrame,
    resolution: float = 1,
    graph_method: str = "umap",
    n_neighbors: int = 15,
    cluster_method: str = "leiden",
):
    assert data.index.equals(metadata.index)
    logger.info(f"cluster data with {len(data)} observations")
    logger.info(f"compute graph with method `{graph_method}`")

    # %%
    if graph_method == "umap":
        min_dist = 0.1
        metric = "euclidean"
        n_components = 2

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
        csr = kneighbors_graph(data, n_neighbors=n_neighbors, mode="distance", metric=metric, include_self=False)
        # assert False, 'we need to convert distance to weights'
    elif graph_method == "scanpy":
        from anndata import AnnData

        if cuda.is_available():
            logger.info("GPU available")
            import rapids_singlecell as sc
        else:
            import scanpy as sc

        # %% create AnnData
        ad = AnnData(X=data.values, obs=metadata)

        if cuda.is_available():
            sc.get.anndata_to_GPU(ad)
        sc.pp.neighbors(ad, n_neighbors=n_neighbors, use_rep="X")
        csr = ad.obsp["connectivities"]

    G = csr_to_ig(csr)

    if cluster_method == "leiden":
        membership = get_leiden_membership(G, resolution=resolution)
    elif cluster_method == "igraph":
        membership = get_igraph_membership(G, resolution=resolution)
    elif cluster_method == "kmeans":
        n_clusters = 20
        membership = get_kmeans_membership(data, n_clusters=n_clusters)
    elif cluster_method == "scanpy":
        assert graph_method == "scanpy"
        sc.tl.leiden(ad, resolution=resolution)
        # sc.tl.leiden(ad, resolution=resolution, flavor="igraph", n_iterations=2)
        membership = ad.obs["leiden"].astype(str).values

    membership = pd.Categorical([str(i) for i in membership])

    num_cluster = np.unique(membership).size
    metadata = metadata.assign(membership=membership)
    result = dict(num_obs=len(data), num_cluster=num_cluster)

    logger.info("completed")
    return metadata


def csr_to_ig(csr: scipy.sparse.csr_matrix, directed=False, weighted=True):
    rows, cols = csr.nonzero()

    g = ig.Graph(directed=directed)
    g.add_vertices(csr.shape[0])
    g.add_edges(list(zip(rows, cols)))

    if weighted:
        weights = csr.data
    else:
        weights = np.ones_like(csr.data)

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
    return G.community_leiden(objective_function="modularity", resolution=resolution, weights="weight").membership


def get_kmeans_membership(data, n_clusters: int = 20):
    from sklearn.cluster import KMeans

    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(data)
    return kmeans.labels_

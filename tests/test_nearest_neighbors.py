import numpy as np
from ai4bmr_learn.clustering.cluster import compute_knn_graph

data = np.random.rand(50_000, 50)
data = [*data, np.random.rand(2, 50) * 16]
data = [*data, np.random.rand(2, 50) * 32]
data = np.vstack(data)

metric = 'euclidean'
n_neighbors = 15
mode = 'distance'

csr = compute_knn_graph(data, n_neighbors=n_neighbors, metric=metric, engine='cuml', algorithm='brute', mode='distance')
indices = csr.indices.reshape(-1, 15)
distances = csr.data.reshape(-1, 15)

assert (indices[:, 0] == np.array(range(len(data)))).all()
distances[:, 0].max()

filter_ = distances[:, 0] > 0
distances[filter_]

from sklearn.neighbors import NearestNeighbors

kwargs = {'n_neighbors': n_neighbors, 'metric': metric}
nn = NearestNeighbors(**kwargs)
nn.fit(data)
csr = nn.kneighbors_graph(data, mode=mode)

distances, indices = nn.kneighbors(data)

indices = csr.indices.reshape(-1, 15)
distances = csr.data.reshape(-1, 15)
distances[:, 0].max()

# %%
from cuml.neighbors import NearestNeighbors
import cupy as cp
import numpy as np

for _ in range(100):
    data = np.random.rand(50_000, 50)
    data = [*data, np.random.rand(1, 50) * 16]
    data = [*data, np.random.rand(1, 50) * 32]
    data = np.vstack(data)

    data = cp.asarray(data)

    metric = 'euclidean'
    n_neighbors = 15
    mode = 'distance'
    algorithm = 'brute'
    kwargs = {'n_neighbors': n_neighbors, 'metric': metric, 'algorithm': algorithm}

    nn = NearestNeighbors(**kwargs)
    nn.fit(data)

    distances, indices = nn.kneighbors(data)
    assert distances[:, 0].max() == 0

filter_ = distances[:, 0] > 0
distances[filter_]

# %%
from umap.umap_ import nearest_neighbors

indices, distances, index = nearest_neighbors(data, n_neighbors=n_neighbors, metric=metric, metric_kwds=None,
                                              angular=False, random_state=None)

assert (indices[:, 0] == np.array(range(len(data)))).all()
distances[:, 0].max()

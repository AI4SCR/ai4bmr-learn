from cuml.neighbors import NearestNeighbors
import cupy as cp
import numpy as np

for n in [1_000]:
    for i in range(8, 32, 2):
        for j in range(8, 32):
            rng = np.random.default_rng(0)
            data = rng.uniform(0, 1, size=(n, i))
            data = [*data, rng.uniform(1, j, size=(1, i))] # this is not stable
            # data = [*data, np.array([j] * i)]  # this is stable
            # data = [*data, np.array([j] * (i // 2) + [0] * (i // 2))]  # this is stable
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
            if not distances[:, 0].max() == 0:
                # indices = np.flatnonzero(distances[:, 0] != 0)
                print(f'num_obs: {n}, feat_dim: {i}, noise magnitude: {j}')
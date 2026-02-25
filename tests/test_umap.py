def test_run_umap():
    import time
    from loguru import logger
    import numpy as np
    from matplotlib import pyplot as plt
    from ai4bmr_learn.plotting.umap import run_umap
    from ai4bmr_learn.plotting.utils import get_grid_dims

    data = np.random.rand(500_000, 50)
    data = [*data, np.random.rand(2, 50) * 16]
    data = [*data, np.random.rand(2, 50) * 32]
    data = np.vstack(data)

    engines = ['cuml', 'umap-learn', 'rapids-sc']
    nrows, ncols = get_grid_dims(len(engines))

    fig, axs = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 5))
    for engine, ax in zip(engines, axs.flat):
        logger.info(f"Running UMAP with engine: {engine}")

        start_time = time.time()
        out = run_umap(data, n_neighbors=15, min_dist=0.3, metric='euclidean', engine=engine)
        end_time = time.time()

        if engine == 'rapids-sc':
            import scanpy as sc
            sc.pl.umap(out, ax=ax, show=False)
        else:
            import umap.plot
            umap.plot.points(out, ax=ax)

        ax.set_title(f'{engine}: Completed in {end_time - start_time:.2f} seconds.')

    for ax in axs.flat[len(engines):]:
        ax.axis('off')

    # fig.tight_layout()
    fig.show()
    plt.close(fig)

def test_run_umap_with_pre_computed_graph():
    from loguru import logger
    import time
    import numpy as np
    from matplotlib import pyplot as plt
    from ai4bmr_learn.plotting.umap import run_umap
    from ai4bmr_learn.plotting.utils import get_grid_dims
    from ai4bmr_learn.clustering.cluster import kneighbors_graph

    data = np.random.rand(500_000, 50)
    data = [*data, np.random.rand(2, 50) * 16]
    data = [*data, np.random.rand(2, 50) * 32]
    data = np.vstack(data)

    n_neighbors = 15
    metric = 'euclidean'
    algorithm = 'brute'
    mode = 'distance'

    csr = kneighbors_graph(data, n_neighbors=n_neighbors, metric=metric, engine='cuml', algorithm=algorithm, mode=mode)
    indices = csr.indices.reshape(-1, 15)
    distances = csr.data.reshape(-1, 15)
    distances[:, 0].max()

    precomputed_knn = (indices, distances)
    umap_kwargs = dict(precomputed_knn=precomputed_knn)

    engines = ['umap-learn']
    nrows, ncols = get_grid_dims(len(engines))

    fig, axs = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 5), squeeze=False)
    for engine, ax in zip(engines, axs.flat):
        logger.info(f"Running UMAP with engine: {engine}")

        start_time = time.time()
        out = run_umap(data, n_neighbors=n_neighbors, min_dist=0.3, metric=metric, engine=engine, **umap_kwargs)
        end_time = time.time()

        if engine == 'rapids-sc':
            import scanpy as sc
            sc.pl.umap(out, ax=ax, show=False)
        else:
            import umap.plot
            umap.plot.points(out, ax=ax)

        ax.set_title(f'{engine}: Completed in {end_time - start_time:.2f} seconds.')

    for ax in axs.flat[len(engines):]:
        ax.axis('off')

    # fig.tight_layout()
    fig.show()
    plt.close(fig)

def test_plot_umap():
    from loguru import logger
    import numpy as np
    from matplotlib import pyplot as plt
    from ai4bmr_learn.plotting.umap import plot_umap
    from ai4bmr_learn.plotting.utils import get_grid_dims
    from ai4bmr_learn.clustering.cluster import compute_knn_graph

    data = np.random.rand(50_000, 50)
    data = [*data, np.random.rand(2, 50) * 16]
    data = [*data, np.random.rand(2, 50) * 32]
    data = np.vstack(data)

    csr = compute_knn_graph(data, n_neighbors=15, metric='euclidean', engine='cuml', algorithm='brute', connectivity='distances', include_self=True)
    umap_kwargs = dict(precomputed_knn=csr)

    engines = ['cuml', 'umap-learn', 'rapids-sc']
    nrows, ncols = get_grid_dims(len(engines))

    fig, axs = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 5))
    for engine, ax in zip(engines, axs.flat):
        logger.info(f"Running UMAP with engine: {engine}")

        _ = plot_umap(data, ax=ax, n_neighbors=15, min_dist=0.3, metric='euclidean', engine=engine, umap_kwargs=umap_kwargs)
        ax.set_title(f'{engine}')

    for ax in axs.flat[len(engines):]:
        ax.axis('off')

    # fig.tight_layout()
    fig.show()
    plt.close(fig)

test_plot_umap()
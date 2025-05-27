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

        case _:
            raise ValueError(f"Unknown engine: {engine}")

def test_run_umap(random_state=None):
    import time
    from loguru import logger
    import numpy as np
    from matplotlib import pyplot as plt

    rng = np.random.default_rng(random_state)
    data = rng.uniform(0,1, (50_000, 50))

    data_noisy = [*data, rng.uniform(0,16,(2, 50))]
    data_noisy = [*data, rng.uniform(0,32,(2, 50))]
    data_noisy = np.vstack(data_noisy)

    engines = ['cuml', 'umap-learn']
    nrows, ncols = 2, 2

    fig, axs = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 5))
    for name, dat, axs_ in zip(['uniform', 'noisy'], [data, data_noisy], axs):
        for engine, ax in zip(engines, axs_):
            logger.info(f"Running UMAP with engine: {engine}")

            start_time = time.time()
            out = run_umap(dat, n_neighbors=15, min_dist=0.3, metric='euclidean', engine=engine, random_state=random_state)
            end_time = time.time()

            import umap.plot
            umap.plot.points(out, ax=ax)

            ax.set_title(f'Data: {name}, Engine: {engine}, RandomState: {random_state}\nRuntime:{end_time - start_time:.2f} seconds.')

    for ax in axs.flat:
        ax.axis('off')

    fig.tight_layout()
    fig.show()
    # fig.savefig(f'/work/FAC/FBM/DBC/mrapsoma/prometex/projects/ai4bmr-learn/issues/random-seed={random_state}.png', dpi=300)
    plt.close(fig)

for _ in range(5):
    test_run_umap(random_state=None)
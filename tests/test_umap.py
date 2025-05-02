from loguru import logger

def test_run_umap():
    import numpy as np
    from matplotlib import pyplot as plt
    from ai4bmr_learn.plotting.umap import run_umap
    from ai4bmr_core.utils.plotting import get_grid_dims

    data = np.random.rand(50_000, 50)
    data = [*data, np.random.rand(2, 50) * 16]
    data = [*data, np.random.rand(2, 50) * 32]
    data = np.vstack(data)

    engines = ['cuml', 'umap-learn', 'rapids-sc']
    nrows, ncols = get_grid_dims(len(engines))

    fig, axs = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 5))
    for engine, ax in zip(engines, axs.flat):
        logger.info(f"Running UMAP with engine: {engine}")

        out = run_umap(data, n_neighbors=15, min_dist=0.3, metric='euclidean', engine=engine)

        if engine == 'rapids-sc':
            import scanpy as sc
            sc.pl.umap(out, ax=ax, show=False)
        else:
            import umap.plot
            umap.plot.points(out, ax=ax)

        ax.set_title(f'{engine}')

    for ax in axs.flat[len(engines):]:
        ax.axis('off')

    # fig.tight_layout()
    fig.show()
    plt.close(fig)
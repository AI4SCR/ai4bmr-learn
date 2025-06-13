
if __name__ == "__main__":
    from ai4bmr_learn.datamodules.image_embeddings import ImageEmbeddings
    from ai4bmr_learn.routines.estimator_cv import estimator_cv, SweepConfig, WandbInitConfig
    from jsonargparse import ArgumentParser
    import os

    parser = ArgumentParser()
    parser.add_argument("--config", action="config")
    parser.add_class_arguments(ImageEmbeddings, 'datamodule')

    args = parser.parse_args()

    datamodule = parser.instantiate_classes(args).datamodule

    sweep = SweepConfig()

    wandb_init = WandbInitConfig(project="image-features")
    os.environ["WANDB_API_KEY"] = '0869e6d0e018860ee017240132100bfd828c056a'

    estimator_cv(
        datamodule=datamodule,
        sweep=sweep,
        wandb_init=wandb_init,
        metrics=None
    )
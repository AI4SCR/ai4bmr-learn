import dotenv
dotenv.load_dotenv()

if __name__ == "__main__":
    from ai4bmr_learn.datamodules.image_embedding import ImageEmbedding
    from ai4bmr_learn.routines.logistic_regression import logistic_regression, SweepConfig, WandbInitConfig

    from jsonargparse import ArgumentParser
    import os

    parser = ArgumentParser()
    parser.add_argument("--config", action="config")
    parser.add_class_arguments(ImageEmbedding, 'datamodule')

    args = parser.parse_args()

    datamodule = parser.instantiate_classes(args).datamodule

    sweep = SweepConfig()

    wandb_init = WandbInitConfig(project="image-features")
    os.environ["WANDB_API_KEY"] = os.environ['WANDB_API_KEY_ETHZ']

    logistic_regression(
        datamodule=datamodule,
        sweep=sweep,
        wandb_init=wandb_init,
        metrics=None
    )
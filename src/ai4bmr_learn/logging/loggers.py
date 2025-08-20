from lightning.pytorch.loggers import wandb
import os

from dotenv import load_dotenv
load_dotenv()

class WandbLogger(wandb.WandbLogger):

    def __init__(
        self,
        api_key_env_var: str | None = None,
        base_url_env_var: str | None = None,
        **kwargs,
    ):
        # Set wandb-related env variables from the provided *env var names*
        if api_key_env_var is not None:
            os.environ["WANDB_API_KEY"] = os.environ[api_key_env_var]
        if base_url_env_var is not None:
            os.environ["WANDB_BASE_URL"] = os.environ[base_url_env_var]

        super().__init__(**kwargs)
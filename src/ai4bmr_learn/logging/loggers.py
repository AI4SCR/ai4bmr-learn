from lightning.pytorch.loggers import wandb
import os

# from dotenv import load_dotenv
# load_dotenv()

class WandbLogger(wandb.WandbLogger):

    def __init__(self, api_key_name: str | None = None, base_url_nameL: str | None = None, **kwargs):
        if api_key_name is not None:
            os.environ['WANDB_API_KEY'] = os.environ[api_key_name]
        if base_url_nameL is not None:
            os.environ['WANDB_BASE_URL'] = os.environ[api_key_name]

        super.__init__(**kwargs)
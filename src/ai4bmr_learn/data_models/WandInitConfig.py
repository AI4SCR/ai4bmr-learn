from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class WandbInitConfig:
    project: str = "default"
    entity: str = "chuv"
    name: str = None
    tags: list[str] = None
    notes: str = ""
    mode: str = "online"
    dir: Path = Path("/work/FAC/FBM/DBC/mrapsoma/prometex/logs/wandb").expanduser().resolve()
    config: dict = field(default_factory=dict)

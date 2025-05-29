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


@dataclass
class ProjectConfig:
    name: str = "default"
    base_log_dir: Path = Path("/work/FAC/FBM/DBC/mrapsoma/prometex/logs/")
    base_ckpt_dir: Path = Path("/work/FAC/FBM/DBC/mrapsoma/prometex/ckpt/")

    @property
    def log_dir(self) -> Path:
        return self.base_log_dir / self.name

    @property
    def ckpt_dir(self) -> Path:
        return self.base_ckpt_dir / self.name


@dataclass
class TrainerConfig:
    max_epochs: int = 100
    accumulate_grad_batches: int = 1
    log_every_n_steps: int = 50
    precision: str | int = "32-true"
    accelerator: str = "auto"
    gradient_clip_val: float | None = None
    devices: int = 1
    fast_dev_run: bool = False


@dataclass
class TrainingConfig:
    ckpt_path: Path | None = None

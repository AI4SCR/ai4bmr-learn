import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from glom import glom
from ai4bmr_learn.utils.pooling import pool
from loguru import logger


class ClipLit(L.LightningModule):
    def __init__(
            self,
            *,
            encoder_1: nn.Module,
            encoder_1_dim: int,
            encoder_2: nn.Module,
            encoder_2_dim: int,
            embed_dim: int,
            encoder_1_pool: str | None = None,
            encoder_2_pool: str | None = None,
            mod_1_key: str = "modalities.image",
            mod_2_key: str = "modalities.expr_tokens",
            lr: float = 1e-5,
            weight_decay: float = 0.01,
            eta: float = 0.0,
            schedule: str | None = None,
            max_epochs: int = 1000,
            num_warmup_epochs: int = 10,
            logit_scale_init: float = 2.6592,
            save_hparams: bool = False,
    ):
        super().__init__()

        self.encoder_1 = encoder_1
        self.encoder_2 = encoder_2

        self.proj_1 = nn.Linear(encoder_1_dim, embed_dim, bias=False)
        self.proj_2 = nn.Linear(encoder_2_dim, embed_dim, bias=False)

        self.logit_scale = nn.Parameter(torch.tensor(float(logit_scale_init)), requires_grad=True)

        self.mod_1_key = mod_1_key
        self.mod_2_key = mod_2_key

        self.encoder_1_pool = encoder_1_pool
        self.encoder_2_pool = encoder_2_pool

        self.lr = lr
        self.weight_decay = weight_decay
        self.eta = eta
        self.schedule = schedule
        self.max_epochs = max_epochs
        self.num_warmup_epochs = num_warmup_epochs

        if save_hparams:
            self.save_hyperparameters(ignore=["encoder_1", "encoder_2"])

    def forward(self, mod1: torch.Tensor, mod2: torch.Tensor):
        f1 = self.encoder_1(mod1)
        f1 = pool(f1, strategy=self.encoder_1_pool)

        f2 = self.encoder_2(mod2)
        f2 = pool(f2, strategy=self.encoder_2_pool)

        if f1.ndim != 2 or f2.ndim != 2:
            raise ValueError(f"Expected pooled features [B, D]. Got f1={tuple(f1.shape)}, f2={tuple(f2.shape)}")

        e1 = F.normalize(self.proj_1(f1), dim=-1)
        e2 = F.normalize(self.proj_2(f2), dim=-1)
        return e1, e2

    def _clip_loss(self, e1: torch.Tensor, e2: torch.Tensor):
        logit_scale = self.logit_scale.exp().clamp(max=100.0)
        logits_12 = logit_scale * (e1 @ e2.t())  # [B,B]
        logits_21 = logits_12.t()

        targets = torch.arange(e1.shape[0], device=e1.device)
        loss_12 = F.cross_entropy(logits_12, targets)
        loss_21 = F.cross_entropy(logits_21, targets)

        return 0.5 * (loss_12 + loss_21), logits_12, logits_21

    @staticmethod
    def _retrieval_metrics(logits: torch.Tensor, k: int) -> torch.Tensor:
        """
        logits: [B,B], where correct match for row i is column i
        returns recall@k in [0,1]
        """
        b = logits.shape[0]
        targets = torch.arange(b, device=logits.device)

        # topk indices per row: [B, k]
        topk = logits.topk(k=min(k, b), dim=1).indices
        hit = (topk == targets[:, None]).any(dim=1)
        return hit.float().mean()

    def _shared_step(self, batch: dict, stage: str):
        mod1 = glom(batch, self.mod_1_key)
        mod2 = glom(batch, self.mod_2_key)

        e1, e2 = self(mod1, mod2)
        loss, logits_12, logits_21 = self._clip_loss(e1, e2)

        bs = int(e1.shape[0])

        with torch.no_grad():
            # retrieval metrics (in-batch)
            r1_12 = self._retrieval_metrics(logits_12, k=1)
            r5_12 = self._retrieval_metrics(logits_12, k=5)
            r1_21 = self._retrieval_metrics(logits_21, k=1)
            r5_21 = self._retrieval_metrics(logits_21, k=5)

            r1 = 0.5 * (r1_12 + r1_21)
            r5 = 0.5 * (r5_12 + r5_21)

            # collapse diagnostics
            # norms should be ~1 due to normalize
            e1_norm = e1.norm(dim=1).mean()
            e2_norm = e2.norm(dim=1).mean()
            e1_std = e1.std(dim=0).mean()
            e2_std = e2.std(dim=0).mean()

            # diag vs off-diag similarities (use logits_12 without scale to be interpretable)
            sims = e1 @ e2.t()  # cosine sims because normalized
            diag = sims.diag()
            diag_mean = diag.mean()

            mask = ~torch.eye(bs, dtype=torch.bool, device=sims.device)
            off = sims[mask]
            off_mean = off.mean()
            off_std = off.std()

            gap = diag_mean - off_mean

        # core logs
        self.log(f"loss/{stage}", loss, on_step=(stage == "train"), on_epoch=True, batch_size=bs)
        self.log("logit_scale_exp", self.logit_scale.exp(), on_step=(stage == "train"), on_epoch=True, batch_size=bs)

        # retrieval logs (averaged across directions)
        self.log(f"{stage}/recall@1", r1, on_step=False, on_epoch=True, batch_size=bs)
        self.log(f"{stage}/recall@5", r5, on_step=False, on_epoch=True, batch_size=bs)

        # optionally also per-direction (useful for debugging modality imbalance)
        self.log(f"{stage}/recall@1_12", r1_12, on_step=False, on_epoch=True, batch_size=bs)
        self.log(f"{stage}/recall@1_21", r1_21, on_step=False, on_epoch=True, batch_size=bs)

        # collapse monitoring logs
        self.log(f"{stage}/e1_norm", e1_norm, on_step=False, on_epoch=True, batch_size=bs)
        self.log(f"{stage}/e2_norm", e2_norm, on_step=False, on_epoch=True, batch_size=bs)
        self.log(f"{stage}/e1_std", e1_std, on_step=False, on_epoch=True, batch_size=bs)
        self.log(f"{stage}/e2_std", e2_std, on_step=False, on_epoch=True, batch_size=bs)

        self.log(f"{stage}/sim_diag_mean", diag_mean, on_step=False, on_epoch=True, batch_size=bs)
        self.log(f"{stage}/sim_off_mean", off_mean, on_step=False, on_epoch=True, batch_size=bs)
        self.log(f"{stage}/sim_off_std", off_std, on_step=False, on_epoch=True, batch_size=bs)
        self.log(f"{stage}/sim_gap", gap, on_step=False, on_epoch=True, batch_size=bs)

        return loss

    def on_train_start(self) -> None:
        optimizer = self.optimizers()
        params_in_optimizer = [id(p) for group in optimizer.param_groups for p in group["params"]]
        assert id(self.logit_scale) in params_in_optimizer, (
            f"logit_scale (id: {id(self.logit_scale)}) is NOT in the optimizer!"
        )

    def on_train_epoch_start(self) -> None:
        # NOTE: we freeze the logit_scale during warmup
        self.logit_scale.requires_grad = (self.current_epoch >= self.num_warmup_epochs)
        # assert self.logit_scale.requires_grad


    def training_step(self, batch, batch_idx: int):
        return self._shared_step(batch, stage="train")

    def validation_step(self, batch, batch_idx: int):
        self._shared_step(batch, stage="val")

    def test_step(self, batch, batch_idx: int):
        self._shared_step(batch, stage="test")

    def configure_optimizers(self):
        decay = []
        no_decay = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue

            if name == "logit_scale":
                continue

            if len(param.shape) == 1 or name.endswith(".bias"):
                no_decay.append(param)
            else:
                decay.append(param)

        optimizer = optim.AdamW(
            [
                {"params": decay, "weight_decay": self.weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
                {"params": [self.logit_scale], "lr": self.lr, "weight_decay": 0.0},
            ],
            lr=self.lr,
        )

        if self.schedule is None:
            return optimizer

        max_epochs = getattr(self.trainer, "max_epochs", None) or self.max_epochs

        warmup = optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1e-2, end_factor=1.0, total_iters=self.num_warmup_epochs
        )
        t_max = max(1, max_epochs - self.num_warmup_epochs)
        cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=self.eta)

        scheduler = optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, cosine], milestones=[self.num_warmup_epochs]
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch", "frequency": 1},
        }

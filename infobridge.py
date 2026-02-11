import math
from typing import Callable, Optional, Any, Dict

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from pytorch_lightning.callbacks import TQDMProgressBar
import pytorch_lightning as pl
from torch_ema import ExponentialMovingAverage

def generate_derangement(size: int, device: str="cpu") -> torch.Tensor:

    if size <= 1:
        raise ValueError(f"Can't create derangement of size {size}.")

    identity = torch.arange(size, device=device)

    while True:
        permutation = torch.randperm(size, device=device)
        if not (permutation == identity).any():
            return permutation
            
class NoValTQDMProgressBar(TQDMProgressBar):
    def init_validation_tqdm(self):
        bar = super().init_validation_tqdm()
        bar.disable = True
        return bar


class BridgeMatching(nn.Module):
    """
    Bridge matching objective, abstract w.r.t. architecture and data.

    This is a cleaned-up version of `BridgeMathcing` from the notebook.

    The module expects a `vector_net` with signature:

        vector_net(x_t, x_0, t) -> same shape as x_t

    where:
      - x_t: bridge state at time t, shape (B, C, ...)
      - x_0: initial state, shape (B, C, ...)
      - t: scalar or tensor of shape (B,) in [0, 1]
    """

    def __init__(
        self,
        vector_net: nn.Module,
        eps: float,
    ) -> None:
        super().__init__()

        self.vector_net = vector_net
        self.eps = float(eps)

    @staticmethod
    def _broadcast_t(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Broadcast a time tensor of shape (B,) or (B, 1, ..., 1) to match `x`'s
        batch dimension and have singleton entries for all remaining dims.

        Works for arbitrary data shapes:
          - x: (B, D)
          - x: (B, D1, D2)
          - x: (B, D1, D2, D3)
        """
        if t.dim() == 1:
            # Shape: (B, 1, ..., 1) with rank matched to x
            shape = [t.size(0)] + [1] * (x.dim() - 1)
            t = t.view(*shape)
        else:
            # For already batched t, just expand trailing singleton dims
            while t.dim() < x.dim():
                t = t.unsqueeze(-1)
        return t

    def sample_x_t(
        self, x_0: torch.Tensor, x_1: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """
        Sample x_t from the noisy bridge between x_0 and x_1 at time t.
        """
        coef_0, coef_1 = 1.0 - t, t
        t_b = self._broadcast_t(t, x_0)
        coef_0_b = self._broadcast_t(coef_0, x_0)
        coef_1_b = self._broadcast_t(coef_1, x_1)

        std_t = torch.sqrt(t * (1.0 - t) * self.eps)
        std_t_b = self._broadcast_t(std_t, x_0)

        z = torch.randn_like(x_0, device=x_0.device)
        return coef_1_b * x_1 + coef_0_b * x_0 + z * std_t_b

    def loss(
        self,
        x_t_hat: torch.Tensor,
        x_1: torch.Tensor,
        x_0: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Per-sample loss (no reduction).
        """
        b = x_1.shape[0]
        flat = lambda x: x.view(b, -1)

        # Broadcast time to match data shape for all operations below
        t_b = self._broadcast_t(t, x_1)

        target = (x_1 - x_t) / (1.0 - t_b)
        return torch.norm(flat(x_t_hat - target), dim=-1)

    def step(
        self, x_0: torch.Tensor, x_1: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """
        One bridge-matching step for a batch.

        Args:
            x_0: (B, C, ...)
            x_1: (B, C, ...)
            t:  (B,) or (B, 1, 1, 1) with values in (0, 1)
        """
        t_flat = t.view(-1)
        x_t = self.sample_x_t(x_0, x_1, t_flat)
        # vector_net is expected to take (x_t, x_0, t)
        x_t_hat = self.vector_net(x_t, x_0, t_flat)
        return self.loss(x_t_hat, x_1, x_0, x_t, t_flat).mean()


class DriftWrapper(nn.Module):
    """
    Generic wrapper around a user-provided backbone to implement a drift.

    By default this assumes a `diffusers`-style UNet2DModel-like interface:

        backbone(x, t, class_labels=...) -> object with `.sample` tensor

    If your backbone returns tensors directly, `.sample` will not be accessed.
    """

    def __init__(self, backbone: nn.Module, plan_label: float = 1.0) -> None:
        super().__init__()
        self.backbone = backbone
        self.plan_label = float(plan_label)

    def forward(
        self, x_t: torch.Tensor, x_0: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        # Concatenate along feature/channel dimension by default.
        x = torch.cat([x_0, x_t], dim=1)

        # Ensure t is shape (B,) for typical diffusion backbones.
        if t.dim() != 1:
            t = t.view(-1)

        batch_size = x.shape[0]
        class_labels = (
            torch.full(
                (batch_size,),
                self.plan_label,
                dtype=t.dtype,
                device=x.device,
            )
        )

        out = self.backbone(x, t, class_labels=class_labels)
        out_tensor = out.sample if hasattr(out, "sample") else out

        # Reshape backbone output back to the shape of x_t so that the drift
        # matches the data tensor shape (supports [N, D], [N, D1, D2], [N, D1, D2, D3]).
        return out_tensor.view_as(x_t)


@torch.no_grad()
def sample_posterior(
    vector_net: nn.Module,
    eps: float,
    x_0: torch.Tensor,
    nfe: int = 100,
    pbar: bool = False,
) -> torch.Tensor:
    """
    Sample x_1 given x_0 using the learned drift `vector_net`.

    This is a generalized version of the notebook's `sample_posterior`.
    """
    device = x_0.device
    x_t = x_0
    euler_dt = 1.0 / float(nfe)

    t_range = torch.arange(0.0, 1.0, step=euler_dt, device=device)

    for t in t_range:
        t_next = t + euler_dt

        x_1_predict = x_t + (1.0 - t) * vector_net(x_t, x_0, t.expand(x_t.size(0)))

        std = torch.sqrt(eps * (t_next - t) * (1.0 - t_next) / (1.0 - t))
        std_b = std.view(*([1] * x_t.dim())).expand_as(x_t)
        coef = (1.0 - t_next) / (1.0 - t)
        mean_x_t = coef * x_t + (1.0 - coef) * x_1_predict

        x_t = mean_x_t + std_b * torch.randn_like(x_t)

    return x_t

class InfoBridge(pl.LightningModule):
    """
    PyTorch Lightning module that encapsulates bridge-matching-based
    mutual information estimation.

    - Abstract over backbone architecture: user provides a backbone network.
    - Abstract over data: training and MI estimation use input tensors.

    Typical usage:

        model = InfoBridge(backbone, eps=1.0e-3)
        model.fit(x0_train, x1_train, max_epochs=..., batch_size=...)
        mi_hat = model.estimate_mi(x0_test, x1_test)
    """

    def __init__(
        self,
        backbone: nn.Module,
        eps: float,
        lr: float = 1e-4,
        t_eps: float = 1e-6,
        nfe_posterior: int = 100,
        base_mc_iters: int = 20,
        default_batch_size: int = 128,
        default_max_epochs: int = 1,
        ema_decay: float = 0.999,
        ema_update_every: int = 1,
    ) -> None:
        super().__init__()

        self.save_hyperparameters(ignore=["backbone"])

        self.backbone = backbone
        # Two drift wrappers sharing the same backbone parameters
        self.drift_plan = DriftWrapper(self.backbone, plan_label=1.0)
        self.drift_ind = DriftWrapper(self.backbone, plan_label=-1.0)

        self.bm_plan = BridgeMatching(
            self.drift_plan,
            eps=eps,
        )
        self.bm_ind = BridgeMatching(
            self.drift_ind,
            eps=eps,
        )

        self.lr = float(lr)
        self.eps = float(eps)
        self.t_eps = float(t_eps)
        self.nfe_posterior = int(nfe_posterior)
        self.base_mc_iters = int(base_mc_iters)
        self.default_batch_size = int(default_batch_size)
        self.default_max_epochs = int(default_max_epochs)
        self.ema_decay = float(ema_decay)
        self.ema_update_every = max(1, int(ema_update_every))

        self.ema = ExponentialMovingAverage(
            self.backbone.parameters(),
            decay=self.ema_decay,
        )

    # ------------------------------------------------------------------
    # Lightning hooks
    # ------------------------------------------------------------------
    def configure_optimizers(self):
        # Single optimizer over backbone parameters (shared by both drifts).
        optimizer = torch.optim.AdamW(self.backbone.parameters(), lr=self.lr)
        return optimizer

    def _sample_t(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Sample times t ~ uniform(0, 1 - t_eps].
        """
        t = torch.rand(batch_size, device=device) * (1.0 - self.t_eps)
        return t.view(-1, 1, 1, 1)

    def on_fit_start(self) -> None:
        self.ema.to(device=next(self.backbone.parameters()).device)

    @torch.no_grad()
    def on_train_batch_end(self, outputs, batch, batch_idx: int) -> None:
        if (batch_idx + 1) % self.ema_update_every != 0:
            return
        self.ema.update()

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        checkpoint["ema_state_dict"] = self.ema.state_dict()

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        ema_state = checkpoint.get("ema_state_dict")
        if ema_state is not None:
            self.ema.load_state_dict(ema_state)
            self.ema.to(device=next(self.backbone.parameters()).device)

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        """
        Expects batch = (x0, x1).
        Updates both plan and independent drifts in a single optimizer step.
        """
        x0, x1 = batch
        device = x0.device
        b = x0.size(0)

        t = self._sample_t(b, device)

        loss_plan = self.bm_plan.step(x0, x1, t)

        idx = generate_derangement(b, device)
        
        # idx = torch.randperm(b, device=device)
        
        x1_perm = x1[idx]
        t_ind = self._sample_t(b, device)
        loss_ind = self.bm_ind.step(x0, x1_perm, t_ind)

        loss = loss_plan + loss_ind
        
        self.log("loss_total", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    @torch.no_grad()
    def _estimate_batch_mi(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
    ) -> torch.Tensor:
        """Monte Carlo MI estimate for a single batch."""

        device = next(self.parameters()).device
        x0 = x0.to(device)
        x1 = x1.to(device)
        b = x0.shape[0]

        t = torch.rand(b, device=device) * (1.0 - 2.0 * self.t_eps) + self.t_eps
        shape = [b] + [1] * (x0.dim() - 1)
        t_b = t.view(*shape)

        mean = x1 * t_b + x0 * (1.0 - t_b)
        std = torch.sqrt(self.eps * t * (1.0 - t))
        std_b = std.view(*shape)
        x_t = mean + std_b * torch.randn_like(x1)

        v_plan = self.drift_plan(x_t, x0, t)
        v_ind = self.drift_ind(x_t, x0, t)

        diff = v_plan - v_ind
        return (diff ** 2).sum(dim=tuple(range(1, diff.dim()))).mean() / (2.0 * self.eps)

    def validation_step(self, batch, batch_idx: int) -> torch.Tensor:
        """Log MI estimate on the validation set during training."""
        x0, x1 = batch
        
        device = next(self.parameters()).device
        
        self.ema.to(device=device)
        with self.ema.average_parameters():
            val_mi = self._estimate_batch_mi(x0, x1)
        self.log(
            "val_mi",
            val_mi,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=False,
        )
        return val_mi

    # ------------------------------------------------------------------
    # High-level convenience API
    # ------------------------------------------------------------------
    def forward(self, x0: torch.Tensor) -> torch.Tensor:
        """
        Convenience forward: sample x1 given x0 using the plan drift.
        """
        return sample_posterior(
            self.drift_plan,
            self.eps,
            x0,
            nfe=self.nfe_posterior,
            pbar=False,
        )

    def fit(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        x0_val: Optional[torch.Tensor] = None,
        x1_val: Optional[torch.Tensor] = None,
        batch_size: Optional[int] = None,
        max_epochs: Optional[int] = None,
        num_workers: int = 0,
        trainer_kwargs: Optional[Dict[str, Any]] = None,
    ) -> "InfoBridge":
        """
        Convenience wrapper around `pytorch_lightning.Trainer.fit` that
        accepts raw tensors and constructs DataLoaders internally.

        Args:
            x0, x1: tensors of shape (N, C, ...) on CPU or GPU.
            x0_val, x1_val: optional validation tensors. Provide both to
                enable validation during fit.
            batch_size: training batch size. If None, uses the value from
                `default_batch_size` specified at initialization.
            max_epochs: number of epochs to train. If None, uses the value from
                `default_max_epochs` specified at initialization.
            num_workers: DataLoader workers.
            trainer_kwargs: extra kwargs forwarded to `pl.Trainer`.
        """
        if trainer_kwargs is None:
            trainer_kwargs = {}

        if batch_size is None:
            batch_size = self.default_batch_size
        if max_epochs is None:
            max_epochs = self.default_max_epochs

        device = next(self.parameters()).device

        train_ds = TensorDataset(x0, x1)
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )

        if (x0_val is None) != (x1_val is None):
            raise ValueError("Provide both x0_val and x1_val, or neither.")

        if x0_val is not None and x1_val is not None:
            val_ds = TensorDataset(x0_val, x1_val)
            val_loader = DataLoader(
                val_ds,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
            )
        else:
            val_loader = None

        trainer = pl.Trainer(max_epochs=max_epochs, callbacks=[NoValTQDMProgressBar(refresh_rate=1)], **trainer_kwargs)
        trainer.fit(self, train_loader, val_loader)
        return self

    @torch.no_grad()
    def estimate_mi(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        batch_size: Optional[int] = None,
        mc_iters: Optional[int] = None,
        num_workers: int = 0
    ) -> torch.Tensor:
        """
        Estimate mutual information between (x0, x1) using the two learned drifts.
        """
        if mc_iters is None:
            mc_iters = self.base_mc_iters
        if batch_size is None:
            batch_size = self.default_batch_size

        # Ensure data is on the same device as the model.
        device = next(self.parameters()).device
        x0 = x0.to(device)
        x1 = x1.to(device)
        
        dataset = TensorDataset(x0, x1)

        dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
            )

        self.drift_plan.eval()
        self.drift_ind.eval()

        self.ema.to(device=device)
        with self.ema.average_parameters():
            mi_value = torch.tensor(0.0, device=device)
            total_batches = 0
            for _ in range(mc_iters):
                for x0_batch, x1_batch in dataloader:
                    mi_value += self._estimate_batch_mi(
                        x0_batch,
                        x1_batch,
                    )
                    total_batches += 1
            return mi_value / total_batches

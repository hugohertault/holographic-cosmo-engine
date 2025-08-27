# AdS5 Klein-Gordon PINN minimal solver (fixed AdS background, spherically symmetric)
# - Equation (radial, 5D): phi'' + (4/r) phi' - (m^2/L^2) phi - (lambda/L^2) phi^3 = 0
# - Asymptotic (r -> inf): phi(r) ~ A * r^{-Delta_+}, with Delta_+ = 2 + sqrt(4 + m^2 L^2)
# - We encode phi(r) = phi0 + r^2 * g(r) to ensure phi'(0)=0 automatically.
# - Loss = PDE residual (collocation) + tail penalty near r_max + small center penalty.
# This is intended as a robust smoke-test building block.

from dataclasses import dataclass
from pathlib import Path
import math
import torch
from torch import nn, autograd
import numpy as np
import matplotlib.pyplot as plt

# ---- Config ----
@dataclass
class EKGConfig:
    r_max: float = 8.0
    n_colloc: int = 512
    m2L2: float = -2.5      # m^2 L^2  (BF bound is -4)
    lamL2: float = 0.0      # lambda L^2
    L: float = 1.0
    w_tail: float = 1.0
    w_center: float = 1e-4
    lr: float = 2e-3
    iters_adam: int = 400
    seed: int = 1234

# ---- MLP ----
class MLP(nn.Module):
    def __init__(self, dim_in=1, dim_out=1, width=64, depth=4, act=nn.Tanh):
        super().__init__()
        layers = [nn.Linear(dim_in, width), act()]
        for _ in range(depth-1):
            layers += [nn.Linear(width, width), act()]
        layers += [nn.Linear(width, dim_out)]
        self.net = nn.Sequential(*layers)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    def forward(self, x):
        return self.net(x)

# ---- PINN for scalar with center-safe ansatz ----
class PINNScalar(nn.Module):
    def __init__(self, width=64, depth=4):
        super().__init__()
        self.mlp = MLP(1, 1, width, depth, nn.Tanh)
        # trainable phi0 to shift center value
        self.phi0 = nn.Parameter(torch.tensor(0.0))
    def forward(self, r):
        # r: (N,1) tensor (requires_grad True for PDE)
        g = self.mlp(r)
        # encode phi(r) = phi0 + r^2 * g(r) => phi'(0)=0
        return self.phi0 + (r**2) * g

# ---- PDE residual ----
def KG_residual(phi, r, m2L2=-2.5, lamL2=0.0, L=1.0):
    # phi: (N,1), r: (N,1) with requires_grad True
    dphi_dr = autograd.grad(
        phi, r, grad_outputs=torch.ones_like(phi),
        create_graph=True, retain_graph=True, allow_unused=False
    )[0]
    d2phi_dr2 = autograd.grad(
        dphi_dr, r, grad_outputs=torch.ones_like(dphi_dr),
        create_graph=True, retain_graph=True, allow_unused=False
    )[0]
    # avoid division by zero at r=0 (we do not include exactly r=0 in PDE set)
    res = d2phi_dr2 + (4.0/(r+1e-9)) * dphi_dr - (m2L2/(L**2))*phi - (lamL2/(L**2))*(phi**3)
    return res

# ---- Tail boundary penalty (AdS falloff) ----
def tail_penalty(phi_tail, r_tail, m2L2=-2.5, L=1.0):
    Delta_plus = 2.0 + math.sqrt(4.0 + m2L2)  # independent of L (m2 appears as m2L2/L^2)
    # target: r^{Delta+} * phi(r) -> constant  (we penalize it toward 0 for smoke simplicity)
    scaled = (r_tail**Delta_plus) * phi_tail
    return (scaled**2).mean(), Delta_plus

# ---- Small center penalty (optional) ----
def center_penalty(model, eps=1e-3):
    # Evaluate dphi/dr at small radius ~ eps, penalize it to be ~0 (should be small given ansatz)
    r = torch.tensor([[eps]], dtype=torch.float32, requires_grad=True)
    phi = model(r)
    dphi_dr = autograd.grad(phi, r, grad_outputs=torch.ones_like(phi), create_graph=True)[0]
    return (dphi_dr**2).mean()

# ---- Training ----
def train_ekg(cfg: EKGConfig, model: PINNScalar, outdir: Path):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    outdir.mkdir(parents=True, exist_ok=True)

    # collocation points (exclude exact 0 to avoid 1/r)
    r_in = torch.rand(cfg.n_colloc, 1) * (cfg.r_max - 1e-3) + 1e-3
    r_in = r_in.requires_grad_(True)

    # tail points near r_max
    n_tail = max(64, cfg.n_colloc // 4)
    r_tail = torch.rand(n_tail, 1) * (cfg.r_max - 0.8*cfg.r_max) + 0.8*cfg.r_max
    r_tail = r_tail.requires_grad_(True)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    for it in range(1, cfg.iters_adam + 1):
        opt.zero_grad()

        # PDE residual
        phi_in = model(r_in)
        res = KG_residual(phi_in, r_in, m2L2=cfg.m2L2, lamL2=cfg.lamL2, L=cfg.L)
        loss_pde = (res**2).mean()

        # Tail penalty
        phi_tail = model(r_tail)
        loss_tail, Dp = tail_penalty(phi_tail, r_tail, m2L2=cfg.m2L2, L=cfg.L)

        # Center penalty
        loss_center = center_penalty(model) * cfg.w_center

        loss = loss_pde + cfg.w_tail*loss_tail + loss_center
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        opt.step()

        if it % 200 == 0 or it == 1:
            # convert to float for logging
            print(f"[{it:04d}] L={float(loss):.3e} | KG={float(loss_pde):.3e} | "
                  f"tail={float(loss_tail):.3e} | center={float(loss_center):.3e} | Dp={Dp:.3f}")

    # Plot solution
    with torch.no_grad():
        r_plot = torch.linspace(0.0, cfg.r_max, 400).unsqueeze(1)
        phi_plot = model(r_plot).squeeze(1).cpu().numpy()
        r_np = r_plot.squeeze(1).cpu().numpy()

    import matplotlib
    matplotlib.use("Agg")
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(r_np, phi_plot, lw=2)
    ax.set_xlabel("r")
    ax.set_ylabel("phi(r)")
    ax.set_title("AdS5 KG PINN (smoke)")
    fig.tight_layout()
    fig_path = outdir / "phi.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)

    return {
        "loss": float(loss),
        "loss_pde": float(loss_pde),
        "loss_tail": float(loss_tail),
        "loss_center": float(loss_center),
        "Delta_plus": float(2.0 + math.sqrt(4.0 + cfg.m2L2)),
        "figure": str(fig_path)
    }

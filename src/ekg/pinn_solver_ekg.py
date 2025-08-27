
import math
from dataclasses import dataclass
from pathlib import Path
import torch
import torch.nn as nn
import torch.autograd as autograd
import matplotlib.pyplot as plt

def tanh():
    return nn.Tanh()

class MLP(nn.Module):
    def __init__(self, in_dim=1, out_dim=1, width=64, depth=4):
        super().__init__()
        layers = [nn.Linear(in_dim, width), tanh()]
        for _ in range(depth-1):
            layers += [nn.Linear(width, width), tanh()]
        layers += [nn.Linear(width, out_dim)]
        self.net = nn.Sequential(*layers)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    def forward(self, x):
        return self.net(x)

class PINNScalar(nn.Module):
    def __init__(self, width=64, depth=4, learn_y=False):
        super().__init__()
        self.mlp = MLP(1, 1, width, depth)
        self.learn_y = learn_y
        self.phi0 = nn.Parameter(torch.tensor(0.0))
    def forward(self, r, Delta_plus=None):
        # Regular center via r^2 factor: phi = phi0 + r^2 g(r)
        g = self.mlp(r)
        base = self.phi0 + (r**2) * g
        if self.learn_y and (Delta_plus is not None):
            # Learn y with phi = r^{Delta_plus} * y(r).
            # Since base already has r^2, multiply by r^{Delta_plus-2}.
            return (r.clamp_min(1e-6)**(Delta_plus - 2.0)) * base
        return base

@dataclass
class EKGConfig:
    r_max: float = 8.0
    n_colloc: int = 512
    m2L2: float = -2.5
    lamL2: float = 0.0
    L: float = 1.0
    w_tail: float = 1.0
    w_center: float = 1e-6
    lr: float = 2e-3
    iters_adam: int = 400
    seed: int = 1234
    learn_y: bool = False
    lbfgs_iters: int = 200

def Delta_plus(m2L2):
    return 2.0 + math.sqrt(max(0.0, 4.0 + m2L2))

def KG_residual(phi, r, m2L2=-2.5, lamL2=0.0, L=1.0):
    # phi'' + (4/r) phi' - (m^2 + lambda phi^2) phi = 0  (AdS5 schematic)
    dphi = autograd.grad(phi, r, grad_outputs=torch.ones_like(phi),
                         create_graph=True, retain_graph=True, allow_unused=True)[0]
    if dphi is None:
        dphi = torch.zeros_like(phi, requires_grad=True)
    d2phi = autograd.grad(dphi, r, grad_outputs=torch.ones_like(dphi),
                          create_graph=True, retain_graph=True, allow_unused=True)[0]
    if d2phi is None:
        d2phi = torch.zeros_like(phi, requires_grad=True)
    potp = m2L2 * phi + lamL2 * (phi**3)
    return d2phi + (4.0 / r.clamp_min(1e-6)) * dphi - potp

def save_plot_detached(r, phi, out_png):
    import numpy as np
    rp = r.detach().cpu().numpy().ravel()
    ph = phi.detach().cpu().numpy().ravel()
    plt.figure(figsize=(5.0, 3.4))
    plt.plot(rp, ph, lw=2)
    plt.xlabel("r"); plt.ylabel("phi(r)")
    plt.title("AdS5 KG PINN")
    plt.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close()

def train_ekg(cfg: EKGConfig, model: PINNScalar, out_dir=None):
    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    # collocation points
    r = torch.linspace(1e-6, cfg.r_max, cfg.n_colloc, device=device).view(-1,1)
    r.requires_grad_(True)

    dp = Delta_plus(cfg.m2L2)

    def eval_losses():
        phi = model(r, Delta_plus=dp if cfg.learn_y else None)
        R = KG_residual(phi, r, m2L2=cfg.m2L2, lamL2=cfg.lamL2, L=cfg.L)
        loss_pde = (R**2).mean()
        denom = (r**dp).clamp_min(1e-6)
        loss_tail = ((phi / denom)**2).mean()
        loss_center = (model.phi0**2)
        loss = loss_pde + cfg.w_tail*loss_tail + cfg.w_center*loss_center
        return loss, loss_pde, loss_tail, loss_center, phi

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    # Adam warmup
    for it in range(1, cfg.iters_adam+1):
        opt.zero_grad(set_to_none=True)
        loss, lpde, ltail, lcent, phi = eval_losses()
        loss.backward()
        opt.step()
        if it == 1 or it % 200 == 0:
            print(f"[{it:04d}] L={float(loss):.3e} | KG={float(lpde):.3e} | tail={float(ltail):.3e} | center={float(lcent):.3e} | Dp={dp:.3f}")

    # LBFGS polish (optional)
    if cfg.lbfgs_iters > 0:
        lbfgs = torch.optim.LBFGS(model.parameters(), max_iter=cfg.lbfgs_iters,
                                  tolerance_grad=1e-7, tolerance_change=1e-9,
                                  line_search_fn="strong_wolfe")
        def closure():
            lbfgs.zero_grad()
            loss, _, _, _, _ = eval_losses()
            loss.backward()
            return loss
        lbfgs.step(closure)

    # Final evaluation (IMPORTANT: no torch.no_grad() here; we want autograd inside eval_losses)
    loss, lpde, ltail, lcent, phi = eval_losses()

    # Save detached plot
    out_dir = Path(out_dir) if out_dir else Path("enhanced_runs/quick_smoke_ekg")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_path = out_dir/"phi.png"
    save_plot_detached(r, phi, fig_path)

    return {
        "loss": float((lpde + cfg.w_tail*ltail + cfg.w_center*lcent).item()),
        "loss_pde": float(lpde.item()),
        "loss_tail": float(ltail.item()),
        "loss_center": float(lcent.item()),
        "Delta_plus": float(dp),
        "figure": str(fig_path)
    }

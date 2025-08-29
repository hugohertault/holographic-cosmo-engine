
import math
from dataclasses import dataclass
from pathlib import Path
import torch
from torch import nn, autograd

def Delta_plus_from_m2L2(m2L2: float) -> float:
    # d=4 boundary (AdS5 bulk): Δ± = 2 ± sqrt(4 + m^2 L^2)
    return 2.0 + math.sqrt(max(0.0, 4.0 + m2L2))

class PINNScalar(nn.Module):
    def __init__(self, width=64, depth=4):
        super().__init__()
        layers = [nn.Linear(1, width), nn.Tanh()]
        for _ in range(depth-1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers += [nn.Linear(width, 1)]
        self.net = nn.Sequential(*layers)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)

    def forward(self, r):
        # even ansatz: phi = r^2 * g(r)  => phi'(0)=0
        g = self.net(r)
        return (r*r)*g

@dataclass
class EKGCfg:
    rmax: float = 8.0
    n_colloc: int = 512
    m2L2: float = -2.5
    lamL2: float = 0.0
    L: float = 1.0
    w_kg: float = 1.0
    w_tail: float = 1.0
    w_center: float = 1.0
    w_ein: float = 0.0
    w_dec: float = 0.0
    iters_adam: int = 600
    lr: float = 5e-4
    device: str = "cpu"
    seed: int = 1234

def d_dr(y, x):
    return autograd.grad(y, x, grad_outputs=torch.ones_like(y), create_graph=True, retain_graph=True)[0]

def KG_residual(phi, r, m2L2, lamL2, L):
    phi_r  = d_dr(phi, r)
    phi_rr = d_dr(phi_r, r)
    rm = torch.clamp(r, min=1e-6)
    Vp = (m2L2/(L*L))*phi + (lamL2/(L*L))*(phi*phi)*phi
    return phi_rr + (4.0/rm)*phi_r - Vp

def train_ekg(cfg: EKGCfg, out_dir: Path = None):
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)
    r = torch.linspace(0.0, cfg.rmax, cfg.n_colloc, device=device).view(-1,1).requires_grad_(True)
    model = PINNScalar(width=64, depth=4).to(device)

    huber = nn.SmoothL1Loss(beta=1.0, reduction="mean")
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    def loss_terms(it):
        phi = model(r)
        R   = KG_residual(phi, r, cfg.m2L2, cfg.lamL2, cfg.L)
        kg  = huber(R, torch.zeros_like(R))
        center = (phi[0] - phi[1]).pow(2).mean()
        Dp  = Delta_plus_from_m2L2(cfg.m2L2)
        dphi_dr = d_dr(phi, r)
        rm  = torch.clamp(r[-1,0], min=1e-6)
        tail = (dphi_dr[-1] - (Dp-4.0)/rm * phi[-1]).pow(2).mean()
        ein = kg.detach()*0.0; dec = kg.detach()*0.0
        return {"kg":kg, "tail":tail, "center":center, "ein":ein, "dec":dec, "Dp":Dp, "phi":phi}

    for it in range(1, cfg.iters_adam+1):
        opt.zero_grad(set_to_none=True)
        parts = loss_terms(it)
        wt = cfg.w_tail * min(1.0, it/200.0)
        total = cfg.w_kg*parts["kg"] + wt*parts["tail"] + cfg.w_center*parts["center"]
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        if it % 200 == 0 or it == 1:
            print(f"[Adam {it:04d}] L={float(total):.3e} | KG={float(parts['kg']):.3e} | "
                  f"tail={float(parts['tail']):.3e} | center={float(parts['center']):.3e} | "
                  f"ein={float(parts['ein']):.3e} | dec={float(parts['dec']):.3e} | Dp={parts['Dp']:.3f}")

    lb = torch.optim.LBFGS(model.parameters(), lr=0.8, max_iter=120, history_size=20, line_search_fn="strong_wolfe")
    def closure():
        lb.zero_grad(set_to_none=True)
        parts = loss_terms(cfg.iters_adam)
        loss = cfg.w_kg*parts["kg"] + cfg.w_tail*parts["tail"] + cfg.w_center*parts["center"]
        loss.backward()
        return loss
    try:
        lb.step(closure)
    except Exception as e:
        print("[LBFGS] warning:", e)

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        with torch.no_grad():
            phi = model(r).detach().cpu().numpy()
        import numpy as np, matplotlib.pyplot as plt
        np.save(out_dir/"r.npy", r.detach().cpu().numpy())
        np.save(out_dir/"phi.npy", phi)
        plt.figure(figsize=(5,3))
        plt.plot(r.detach().cpu().numpy(), phi, lw=2)
        plt.xlabel("r"); plt.ylabel("phi"); plt.title("EKG PINN — phi(r)")
        plt.tight_layout(); plt.savefig(out_dir/"phi.png", dpi=150); plt.close()

    return (r, model(r).detach(), None, None, {"Delta_plus": Delta_plus_from_m2L2(cfg.m2L2)})

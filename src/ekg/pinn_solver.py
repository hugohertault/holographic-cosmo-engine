import math, os
import torch
import torch.nn as nn
from torch.autograd import grad

class EKGConfig:
    def __init__(self, L=1.0, m2L2=-2.5, lamL2=0.1, r_max=5.0, n_coll=1024, lr=1e-3, steps=600):
        self.L = L
        self.m2L2 = m2L2
        self.lamL2 = lamL2
        self.r_max = r_max
        self.n_coll = n_coll
        self.lr = lr
        self.steps = steps

def Delta_plus(m2L2):
    return 2.0 + math.sqrt(4.0 + float(m2L2))

class MLP(nn.Module):
    def __init__(self, nin, nout, layers=3, width=64, act="tanh"):
        super().__init__()
        acts = {"tanh": nn.Tanh(), "relu": nn.ReLU(), "gelu": nn.GELU()}
        self.act = acts.get(act, nn.Tanh())
        blocks = []
        in_dim = nin
        for _ in range(layers):
            blocks.append(nn.Linear(in_dim, width))
            blocks.append(self.act)
            in_dim = width
        blocks.append(nn.Linear(in_dim, nout))
        self.net = nn.Sequential(*blocks)

    def forward(self, x):
        if x.dim() == 1:
            x = x.view(-1, 1)
        return self.net(x)

class PINNScalar(nn.Module):
    # Ansatz: phi(r) = phi0 + r^2 * g(r)
    # This ensures phi'(0)=0 automatically.
    # g(r) is represented by a small MLP.
    def __init__(self, phi0=0.5, layers=3, width=64, act="tanh"):
        super().__init__()
        self.phi0 = nn.Parameter(torch.tensor([phi0], dtype=torch.float32))
        self.mlp  = MLP(1, 1, layers, width, act)

    def forward(self, r):
        if r.dim() == 1:
            r2d = r.view(-1, 1)
        else:
            r2d = r
        return self.phi0 + (r2d**2) * self.mlp(r2d)

def KG_residual(phi, r, m2L2=-2.5, lamL2=0.1, L=1.0):
    dp = Delta_plus(m2L2)
    dphi = grad(phi, r, grad_outputs=torch.ones_like(phi), create_graph=True,
                retain_graph=True, allow_unused=True)[0]
    if dphi is None:
        dphi = torch.zeros_like(r)
    d2phi = grad(dphi, r, grad_outputs=torch.ones_like(dphi), create_graph=True,
                 retain_graph=True, allow_unused=True)[0]
    if d2phi is None:
        d2phi = torch.zeros_like(r)
    eps = 1e-6
    kg = d2phi + (4.0/(r+eps))*dphi - (m2L2/(L*L))*phi - (lamL2/(L*L))*(phi**3)
    return kg

@torch.no_grad()
def asymptotic_tail(phi, r, dp):
    rflat = r.view(-1)
    mask = rflat > (0.7 * rflat.max())
    if mask.sum() == 0:
        return torch.tensor(0.0, device=r.device)
    rr = rflat[mask]
    pp = phi.view(-1)[mask]
    target = (rr + 1e-6).pow(-dp)
    return torch.mean((pp.abs() - target).pow(2))

def train_ekg(cfg, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    r = torch.linspace(0.0, cfg.r_max, cfg.n_coll, device=device, requires_grad=True).view(-1, 1)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    dp = Delta_plus(cfg.m2L2)

    for it in range(1, cfg.steps+1):
        opt.zero_grad()
        phi = model(r)
        R = KG_residual(phi, r, m2L2=cfg.m2L2, lamL2=cfg.lamL2, L=cfg.L)
        loss_pde  = torch.mean(R**2)
        loss_bc   = (phi[-1]**2).mean()
        loss_tail = asymptotic_tail(phi, r, dp)
        loss = loss_pde + loss_bc + loss_tail
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if it==1 or it%200==0:
            print(f"[{it:04d}] L={float(loss):.3e} | KG={float(loss_pde):.3e} | tail={float(loss_tail):.3e} | Δ+={dp:.3f}")

    return {"loss": float(loss.detach()), "loss_pde": float(loss_pde.detach()),
            "loss_bc": float(loss_bc.detach()), "Delta_plus": float(dp)}

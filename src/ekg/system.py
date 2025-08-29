
import math, json
from dataclasses import dataclass
from pathlib import Path
import datetime
import torch
import torch.nn as nn
from torch import autograd

# Small MLPs -------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, nin=1, nout=1, hidden=64, depth=3, act=nn.Tanh):
        super().__init__()
        layers = []
        dim = nin
        for _ in range(depth):
            layers += [nn.Linear(dim, hidden), act()]
            dim = hidden
        layers += [nn.Linear(dim, nout)]
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

# Config ----------------------------------------------------------------
@dataclass
class EKGCfg:
    rmax: float = 8.0
    n_colloc: int = 512
    L: float = 1.0
    m2L2: float = -2.5
    lamL2: float = 0.0
    iters_adam: int = 600
    iters_lbfgs: int = 100
    lr: float = 2e-3
    # loss weights
    w_kg: float = 1.0
    w_tail: float = 5.0
    w_center: float = 1e-3
    w_ein: float = 0.2
    w_nec: float = 0.1
    w_dec: float = 0.1
    # BC / tail
    delta_plus: float = None  # if None, set to 2+sqrt(4+m2L2)
    tail_frac: float = 0.08   # last x% points in tail penalty
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 1234

def delta_plus_from_m2L2(m2L2):
    return 2.0 + math.sqrt(4.0 + m2L2)

# PINN fields: phi(r), f(r), N(r) --------------------------------------
class Fields(nn.Module):
    def __init__(self, L=1.0):
        super().__init__()
        self.L = L
        self.nn_phi = MLP(1,1,hidden=64,depth=3)  # g(r) in phi = phi0 + r^2 g(r)
        self.nn_f   = MLP(1,1,hidden=64,depth=3)  # h_f(r) in f = f_ads + r^2 h_f e^{-r/R}
        self.nn_N   = MLP(1,1,hidden=64,depth=3)  # h_N(r) in N = 1 + r^2 h_N e^{-r/R}
        # learned constants
        self.phi0   = nn.Parameter(torch.tensor(0.05))
        self.inv_R  = nn.Parameter(torch.tensor(0.25))  # 1/R_tail >=0 via softplus

    def forward(self, r):
        # r: (N,1) positive
        # scalar with hard center regularity
        g = self.nn_phi(r)
        phi = self.phi0 + (r**2) * g

        # metric ansatz: asymptotic AdS + decaying deformation
        invR = torch.nn.functional.softplus(self.inv_R) + 1e-6
        decay = torch.exp(-invR * r)
        f_ads = 1.0 + (r/self.L)**2
        f = f_ads + (r**2) * self.nn_f(r) * decay

        # lapse ~ 1 + decaying deformation (kept positive-ish via small shift)
        N = 1.0 + (r**2) * self.nn_N(r) * decay
        return phi, f, N

# Derivative helper (autograd) ------------------------------------------
def d_dr(y, r):
    ones = torch.ones_like(y)
    dydr = autograd.grad(y, r, grad_outputs=ones, create_graph=True, retain_graph=True, allow_unused=True)[0]
    if dydr is None:
        # detach gradient path if unused (should not happen in training)
        dydr = torch.zeros_like(y)
    return dydr

# Physics: potential & residuals ---------------------------------------
def potential(phi, m2L2, lamL2, L):
    V = 0.5*(m2L2/(L*L))*phi*phi + 0.25*(lamL2/(L*L))*phi**4
    dV = (m2L2/(L*L))*phi + (lamL2/(L*L))*phi**3
    return V, dV

def KG_residual(phi, f, N, r, m2L2, lamL2, L):
    # (1 / (N r^3)) d_r [ N r^3 f phi_r ] - dV/dphi = 0
    eps = 1e-8
    phi_r = d_dr(phi, r)
    V, dV = potential(phi, m2L2, lamL2, L)
    A = N * (r**3) * f
    div = d_dr(A * phi_r, r)
    kg = div / (N * (r**3 + eps)) - dV
    return kg

def einstein_residuals(phi, f, N, r, m2L2, lamL2, L):
    """
    Mixed components residuals in 5D for metric:
      ds^2 = -N^2 f dt^2 + f^{-1} dr^2 + r^2 dΩ_3^2
    Use:
      G^t_t = 3*(1 - f)/r^2 - (3/2) f'/r
      G^r_r = 3*(1 - f)/r^2 - (3/2) (f * (ln N)')/r
      Λ = -6/L^2 , units 8πG_5=1
    T^t_t = -ρ, ρ = 1/2 f φ'^2 + V
    T^r_r =  p_r = 1/2 f φ'^2 - V
    Residuals (should be 0):
      E^t_t = G^t_t + Λ - 8πG T^t_t = G^t_t + Λ + ρ
      E^r_r = G^r_r + Λ - 8πG T^r_r = G^r_r + Λ - p_r
    """
    eps = 1e-8
    phi_r = d_dr(phi, r)
    V, _ = potential(phi, m2L2, lamL2, L)

    f_r = d_dr(f, r)
    lnN = torch.log(torch.clamp(N, min=1e-6))
    lnN_r = d_dr(lnN, r)

    rho = 0.5 * f * (phi_r**2) + V
    pr  = 0.5 * f * (phi_r**2) - V

    Gtt_mix = 3.0*(1.0 - f)/((r**2)+eps) - 1.5 * f_r/(r + eps)
    Grr_mix = 3.0*(1.0 - f)/((r**2)+eps) - 1.5 * (f * lnN_r)/(r + eps)
    Lambda_mix = -6.0/(L*L)

    E_tt = Gtt_mix + Lambda_mix + rho
    E_rr = Grr_mix + Lambda_mix - pr
    return E_tt, E_rr, rho, pr

def nec_dec_penalties(rho, pr):
    # NEC: rho + pr >= 0  (for canonical scalar f*phi'^2), penalize negative
    nec_q = rho + pr
    nec_pen = torch.relu(-nec_q)**2

    # DEC: rho - |pr| >= 0 , penalize negative
    dec_q = rho - torch.abs(pr)
    dec_pen = torch.relu(-dec_q)**2
    return nec_pen, dec_pen

# Training ---------------------------------------------------------------
def train_ekg(cfg: EKGCfg, out_dir: Path = None):
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)
    # collocation
    r = torch.linspace(0.0, cfg.rmax, cfg.n_colloc, device=device).unsqueeze(1)
    r.requires_grad_(True)

    fields = Fields(L=cfg.L).to(device)
    opt = torch.optim.Adam(fields.parameters(), lr=cfg.lr)

    if cfg.delta_plus is None:
        Delta_plus = delta_plus_from_m2L2(cfg.m2L2)
    else:
        Delta_plus = cfg.delta_plus

    tail_start = int((1.0 - cfg.tail_frac)*cfg.n_colloc)
    tail_idx = slice(max(0, tail_start), cfg.n_colloc)

    def losses():
        # ensure autograd sees r with requires_grad=True even after any optimizer step
        r_local = r.detach().clone().requires_grad_(True)
        # recompute fields on r_local so all derivatives are w.r.t r_local
        phi, f, N = fields(r_local)
        r_use = r_local
        # ---- original body below, but use r_use instead of r ----
        phi, f, N = fields(r)
        # physics
        kg = KG_residual(phi, f, N, r_use, cfg.m2L2, cfg.lamL2, cfg.L)
        E_tt, E_rr, rho, pr = einstein_residuals(phi, f, N, r_use, cfg.m2L2, cfg.lamL2, cfg.L)
        nec_pen, dec_pen = nec_dec_penalties(rho, pr)

        # center penalties (regularity): phi'(0)=0 already hard; also f'(0)=0, N'(0)=0 soft
        f_r = d_dr(f, r_use); N_r = d_dr(N, r_use)
        center = f_r[0,0]**2 + N_r[0,0]**2

        # tail penalties: Robin for phi (suppress source) and asymptotic AdS for f,N
        r_tail = r_use[tail_idx]
        phi_tail = phi[tail_idx]
        w = torch.pow(r_tail, Delta_plus)
        tail_phi = (w * phi_tail)
        tail_phi = (tail_phi**2).mean()

        f_ads_tail = 1.0 + (r_tail/cfg.L)**2
        tail_f = ((f[tail_idx] - f_ads_tail)/(1.0 + (r_tail/cfg.L)**2)).pow(2).mean()
        tail_N = (N[tail_idx] - 1.0).pow(2).mean()

        # aggregate
        loss_kg  = (kg**2).mean()
        loss_ein = (E_tt**2).mean() + (E_rr**2).mean()
        loss_nec = nec_pen.mean()
        loss_dec = dec_pen.mean()
        loss_tail = tail_phi + 0.5*tail_f + 0.5*tail_N

        total = (cfg.w_kg*loss_kg +
                 cfg.w_ein*loss_ein +
                 cfg.w_tail*loss_tail +
                 cfg.w_center*center +
                 cfg.w_nec*loss_nec +
                 cfg.w_dec*loss_dec)

        parts = {
            "kg": loss_kg,
            "ein": loss_ein,
            "tail": loss_tail,
            "center": center,
            "nec": loss_nec,
            "dec": loss_dec
        }
        return total, parts, (phi, f, N)

    # Adam
    for it in range(1, cfg.iters_adam+1):
        opt.zero_grad(set_to_none=True)
        total, parts, _ = losses()
        total.backward()
        torch.nn.utils.clip_grad_norm_(fields.parameters(), max_norm=5.0)
        opt.step()
        if it % 200 == 0 or it == 1:
            dp = delta_plus_from_m2L2(cfg.m2L2)
            print(f"[Adam {it:04d}] L={float(total):.3e} | KG={float(parts['kg']):.3e} | tail={float(parts['tail']):.3e} | center={float(parts['center']):.3e} | ein={float(parts['ein']):.3e} | nec={float(parts['nec']):.3e} | dec={float(parts['dec']):.3e} | Dp={dp:.3f}")

    # LBFGS (finisher)
    lbfgs = torch.optim.LBFGS(fields.parameters(), max_iter=cfg.iters_lbfgs, line_search_fn="strong_wolfe")

    def closure():
        lbfgs.zero_grad()
        total, parts, _ = losses()
        total.backward()
        return total

    try:
        lbfgs.step(closure)
    except Exception as e:
        print("[LBFGS] warning:", repr(e))

    # Final eval (keep graph; we détach only for saving)
    total, parts, (phi, f, N) = losses()
    out = {
            "loss": float(total),
            "loss_kg": float(parts["kg"]),
            "loss_ein": float(parts["ein"]),
            "loss_tail": float(parts["tail"]),
            "loss_center": float(parts["center"]),
            "loss_nec": float(parts["nec"]),
            "loss_dec": float(parts["dec"]),
            "Delta_plus": float(delta_plus_from_m2L2(cfg.m2L2))
        }

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        r_cpu   = r.detach().cpu().numpy().squeeze()
        phi_cpu = phi.detach().cpu().numpy().squeeze()
        f_cpu   = f.detach().cpu().numpy().squeeze()
        N_cpu   = N.detach().cpu().numpy().squeeze()

        import numpy as np, matplotlib.pyplot as plt
        np.save(out_dir/"r.npy", r_cpu)
        np.save(out_dir/"phi.npy", phi_cpu)
        np.save(out_dir/"f.npy", f_cpu)
        np.save(out_dir/"N.npy", N_cpu)

        plt.figure(figsize=(6,3))
        plt.plot(r_cpu, phi_cpu)
        plt.xlabel("r"); plt.ylabel("phi")
        plt.tight_layout(); plt.savefig(out_dir/"phi.png", dpi=150); plt.close()

        plt.figure(figsize=(6,3))
        plt.plot(r_cpu, f_cpu, label="f")
        plt.plot(r_cpu, 1+(r_cpu/cfg.L)**2, "--", label="f_AdS")
        plt.legend(); plt.xlabel("r"); plt.ylabel("metric f")
        plt.tight_layout(); plt.savefig(out_dir/"f.png", dpi=150); plt.close()

        plt.figure(figsize=(6,3))
        plt.plot(r_cpu, N_cpu, label="N")
        plt.axhline(1.0, ls="--", c="k", lw=1, label="N=1")
        plt.legend(); plt.xlabel("r"); plt.ylabel("lapse N")
        plt.tight_layout(); plt.savefig(out_dir/"N.png", dpi=150); plt.close()

        meta = dict(cfg.__dict__)
        meta.update(out)
        meta["timestamp_utc"] = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        (out_dir/"meta.json").write_text(json.dumps(meta, indent=2))

    return r, phi, f, N, out

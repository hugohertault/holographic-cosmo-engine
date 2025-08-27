import math
import torch
from torch import nn, autograd

# Minimal EKG PINN on static AdS5 background (s-wave, phi=phi(r)).
# Metric used for the PDE residual: f(r)=1+r^2/L^2, ds^2 = -f dt^2 + f^{-1} dr^2 + r^2 dOmega_3^2
# Klein-Gordon: (1/r^3) d_r [ r^3 f phi' ] - m^2 phi - lambda phi^3 = 0
# with m^2 = m2L2/L^2 and lambda = lamL2/L^2.

class EKGConfig:
    def __init__(self,
                 n_colloc=512,
                 L=1.0,
                 m2L2=-2.5,
                 lamL2=0.0,
                 r_max=8.0,
                 lr=2e-3,
                 iters_adam=400,
                 seed=1234,
                 w_tail=1.0,
                 w_center=0.1,
                 w_smooth=0.0,
                 w_DEC=0.0):
        self.n_colloc = int(n_colloc)
        self.L = float(L)
        self.m2L2 = float(m2L2)
        self.lamL2 = float(lamL2)
        self.r_max = float(r_max)
        self.lr = float(lr)
        self.iters_adam = int(iters_adam)
        self.seed = int(seed)
        self.w_tail = float(w_tail)
        self.w_center = float(w_center)
        self.w_smooth = float(w_smooth)
        self.w_DEC = float(w_DEC)

    @property
    def Delta_plus(self):
        # For AdS5: Delta(Delta-4)=m^2 L^2 => Delta = 2 + sqrt(4 + m^2 L^2)
        return 2.0 + math.sqrt(max(0.0, 4.0 + self.m2L2))


class MLP(nn.Module):
    def __init__(self, in_dim=1, out_dim=1, hidden=64, layers=4, act=nn.Tanh):
        super().__init__()
        net = []
        net.append(nn.Linear(in_dim, hidden))
        net.append(act())
        for _ in range(layers-1):
            net.append(nn.Linear(hidden, hidden))
            net.append(act())
        net.append(nn.Linear(hidden, out_dim))
        self.net = nn.Sequential(*net)

    def forward(self, x):
        return self.net(x)


class PINNScalarEKG(nn.Module):
    def __init__(self, hidden=64, layers=4):
        super().__init__()
        self.mlp_phi = MLP(1, 1, hidden=hidden, layers=layers)
        # trainable center value phi0 to encourage regularity at r=0
        self.phi0 = nn.Parameter(torch.tensor(0.1))

    def forward(self, r):
        # r is shape [N,1]
        g = self.mlp_phi(r)
        phi = self.phi0 + (r**2) * g
        return phi


def KG_residual(phi, r, L, m2L2, lamL2):
    # Using static AdS5 background:
    # f = 1 + (r/L)^2
    # d_r ( r^3 f phi' ) / r^3 - (m2L2/L^2) phi - (lamL2/L^2) phi^3
    f = 1.0 + (r / L)**2
    m2 = m2L2 / (L**2)
    lam = lamL2 / (L**2)

    # phi' and phi''
    dphi = autograd.grad(
        phi, r, grad_outputs=torch.ones_like(phi),
        create_graph=True, retain_graph=True, allow_unused=True
    )[0]
    if dphi is None:
        dphi = torch.zeros_like(r)

    d2phi = autograd.grad(
        dphi, r, grad_outputs=torch.ones_like(dphi),
        create_graph=True, retain_graph=True, allow_unused=True
    )[0]
    if d2phi is None:
        d2phi = torch.zeros_like(r)

    # df/dr = 2 r / L^2
    df = 2.0 * r / (L**2)

    # Avoid division by zero at r=0: use safe factor (3/r)*f*dphi -> 3*f*phi_r_over_r
    # With our ansatz, dphi ~ O(r), so dphi/r is finite. We compute safely:
    eps = 1e-8
    dphi_over_r = dphi / (r + eps)

    residual = f * d2phi + 3.0 * f * dphi_over_r + df * dphi - m2 * phi - lam * (phi**3)
    return residual


@torch.no_grad()
def _detached(x):
    return x.detach().cpu().numpy()


def train_ekg(cfg: EKGConfig, model: PINNScalarEKG, device="cpu"):
    torch.manual_seed(cfg.seed)
    model.to(device)

    # Collocation points on [0, r_max]
    r = torch.linspace(0.0, cfg.r_max, cfg.n_colloc, device=device).view(-1,1)
    r.requires_grad_(True)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    for it in range(1, cfg.iters_adam+1):
        opt.zero_grad()

        phi = model(r)
        R = KG_residual(phi, r, L=cfg.L, m2L2=cfg.m2L2, lamL2=cfg.lamL2)
        loss_pde = torch.mean(R**2)

        # Tail (Robin-ish): enforce r^{Delta+} * phi(r_max) ~ 0
        dp = cfg.Delta_plus
        phi_tail = model(torch.tensor([[cfg.r_max]], device=device, requires_grad=True))
        loss_tail = torch.mean(( (cfg.r_max**dp) * phi_tail )**2)

        # Center: small penalty on derivative near 0 (extra safety)
        r_small = torch.tensor([[1e-3]], device=device, requires_grad=True)
        phi_small = model(r_small)
        dphi_small = autograd.grad(phi_small, r_small,
                                   grad_outputs=torch.ones_like(phi_small),
                                   create_graph=True, retain_graph=True, allow_unused=True)[0]
        if dphi_small is None:
            dphi_small = torch.zeros_like(r_small)
        loss_center = torch.mean(dphi_small**2)

        # Optional smoothness near center: encourage small second derivative
        loss_smooth = torch.tensor(0.0, device=device)
        if cfg.w_smooth > 0.0:
            d2phi_small = autograd.grad(dphi_small, r_small,
                                        grad_outputs=torch.ones_like(dphi_small),
                                        create_graph=True, retain_graph=True, allow_unused=True)[0]
            if d2phi_small is None: d2phi_small = torch.zeros_like(r_small)
            loss_smooth = torch.mean(d2phi_small**2)

        # Optional DEC penalty (very light, scalar-only proxy):
        # rho = 0.5*f*phi'^2 + V; pr = 0.5*f*phi'^2 - V; want rho >= |pr|.
        loss_dec = torch.tensor(0.0, device=device)
        if cfg.w_DEC > 0.0:
            f = 1.0 + (r / cfg.L)**2
            dphi = autograd.grad(phi, r, grad_outputs=torch.ones_like(phi),
                                 create_graph=True, retain_graph=True, allow_unused=True)[0]
            if dphi is None: dphi = torch.zeros_like(r)
            V = (cfg.m2L2/(2*cfg.L**2))*phi**2 + (cfg.lamL2/(4*cfg.L**2))*phi**4
            rho = 0.5*f*(dphi**2) + V
            pr  = 0.5*f*(dphi**2) - V
            dec_violation = torch.relu(torch.abs(pr) - rho)
            loss_dec = torch.mean(dec_violation**2)

        loss = loss_pde + cfg.w_tail*loss_tail + cfg.w_center*loss_center \
               + cfg.w_smooth*loss_smooth + cfg.w_DEC*loss_dec

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        opt.step()

        if it % 200 == 0 or it == 1:
            print(f"[{it:04d}] L={float(loss):.3e} | KG={float(loss_pde):.3e} | "
                  f"tail={float(loss_tail):.3e} | center={float(loss_center):.3e} | Dp={dp:.3f}")

    stats = {
        "loss": float(loss),
        "loss_pde": float(loss_pde),
        "loss_tail": float(loss_tail),
        "loss_center": float(loss_center),
        "Delta_plus": float(dp),
    }
    return stats

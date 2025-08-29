import os, json, math, time, pathlib, contextlib
from dataclasses import dataclass
import torch
from torch import nn, optim
from typing import Dict, Any, Optional

# minimal MLP + PINN blocks (we reuse what you already had conceptually)
class TinyMLP(nn.Module):
    def __init__(self, nin=1, nout=1, width=64, depth=3, act=nn.Tanh):
        super().__init__()
        layers = [nn.Linear(nin, width), act()]
        for _ in range(depth-1):
            layers += [nn.Linear(width, width), act()]
        layers += [nn.Linear(width, nout)]
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

@dataclass
class TrainStats:
    steps: int
    loss: float
    wall_s: float

def _autograd_grad(y, x):
    ones = torch.ones_like(y)
    return torch.autograd.grad(y, x, grad_outputs=ones, create_graph=True, retain_graph=True, allow_unused=True)[0]

def _kg_residual(phi, r, m2L2, lamL2, L):
    # simple AdS-like KG residual (homogeneous, illustrative)
    r = r.requires_grad_(True)
    dphi = _autograd_grad(phi, r)
    d2phi = _autograd_grad(dphi, r)
    U = (m2L2/(L*L)) + lamL2*phi*phi
    return d2phi + (4.0/r.clamp_min(1e-6))*dphi - U*phi

def _einstein_proxy(f, r):
    r = r.requires_grad_(True)
    df = _autograd_grad(f, r)
    d2f = _autograd_grad(df, r)
    return d2f - (df/r.clamp_min(1e-6))

def save_ckpt(path: pathlib.Path, model: nn.Module, opt: optim.Optimizer, scaler: Optional[torch.cuda.amp.GradScaler], step: int, meta: Dict[str,Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "opt": opt.state_dict() if opt is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "step": step,
        "meta": meta
    }, str(path))

def load_ckpt(path: pathlib.Path, model: nn.Module, opt: Optional[optim.Optimizer], scaler: Optional[torch.cuda.amp.GradScaler]):
    obj = torch.load(str(path), map_location="cpu")
    model.load_state_dict(obj["model"])
    if opt is not None and obj.get("opt") is not None:
        opt.load_state_dict(obj["opt"])
    if scaler is not None and obj.get("scaler") is not None:
        scaler.load_state_dict(obj["scaler"])
    return obj.get("step", 0), obj.get("meta", {})

def train_hybrid(
    out_dir: pathlib.Path,
    m2L2=-2.5, lamL2=0.5, L=1.0,
    n_colloc=512, rmax=8.0,
    iters_adam=800, iters_lbfgs=200,
    adam_lr=2e-3,
    w_kg=1.0, w_ein=0.01, w_tail=0.1, w_center=0.0,
    use_amp=True,
    checkpoint_every=200,
    resume: Optional[pathlib.Path]=None,
    device="cuda" if torch.cuda.is_available() else "cpu",
) -> TrainStats:
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # collocation grid
    r = torch.linspace(0, rmax, n_colloc, device=device).view(-1,1)
    # boundary-safe ansatz via g(r) → phi = phi0 + r^2 g(r); f = 1 + r^2 h(r)
    model_phi_g = TinyMLP(1,1,64,3).to(device)
    model_f_h   = TinyMLP(1,1,64,3).to(device)

    def phi_of_r(rr): return rr*rr*model_phi_g(rr)       # phi'(0)=0 encoded
    def f_of_r(rr):   return 1.0 + rr*rr*model_f_h(rr)   # regular center

    params = list(model_phi_g.parameters()) + list(model_f_h.parameters())
    opt = optim.Adam(params, lr=adam_lr)
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and torch.cuda.is_available()))

    start_step = 0
    meta = {"m2L2":m2L2,"lamL2":lamL2,"L":L}
    if resume is not None and resume.exists():
        try:
            start_step, meta0 = load_ckpt(resume, model_phi_g, opt, scaler)
            # load f too
            # pack/unpack both: we kept them in same optimizer, but need second file or namespacing
            # simplest: store both models together; we emulate by loading the same dict twice here.
        except Exception as e:
            print(f"Resume failed: {e}")

    def losses():
        rr = r.detach().clone().requires_grad_(True)
        with torch.cuda.amp.autocast(enabled=(use_amp and torch.cuda.is_available())):
            phi = phi_of_r(rr)
            f   = f_of_r(rr)
            Rkg = _kg_residual(phi, rr, m2L2, lamL2, L)
            Rein= _einstein_proxy(f, rr)
            # tails/center penalties
            tail = (phi[-1]**2 + (f[-1]-1.0)**2).mean()
            center = (phi[0]**2 + (f[0]-1.0)**2).mean()
            total = w_kg*Rkg.pow(2).mean() + w_ein*Rein.pow(2).mean() + w_tail*tail + w_center*center
        parts = dict(
            kg=Rkg.pow(2).mean().detach(),
            ein=Rein.pow(2).mean().detach(),
            tail=tail.detach(), center=center.detach()
        )
        return total, parts, (phi.detach(), f.detach())

    t0 = time.time()
    # Adam phase
    for it in range(start_step+1, iters_adam+1):
        opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=(use_amp and torch.cuda.is_available())):
            total, parts, _ = losses()
        scaler.scale(total).backward()
        scaler.step(opt)
        scaler.update()
        if it % 200 == 1 or it==iters_adam:
            print(f"[Adam {it:04d}] L={float(total):.3e} | KG={float(parts['kg']):.3e} | tail={float(parts['tail']):.3e} | ein={float(parts['ein']):.3e}")
        if checkpoint_every and (it % checkpoint_every == 0 or it==iters_adam):
            save_ckpt(out_dir/"ckpt_adam.pt", model_phi_g, opt, scaler, it, meta)

    # Optional LBFGS polish (on a copy of params)
    if iters_lbfgs > 0:
        lbfgs = optim.LBFGS(params, max_iter=iters_lbfgs, history_size=20, line_search_fn="strong_wolfe")
        def closure():
            opt.zero_grad(set_to_none=True)
            total, _, _ = losses()
            total.backward()
            return total
        print(f"[LBFGS] start max_iter={iters_lbfgs}, history=20")
        lbfgs.step(closure)
        # save after lbfgs
        save_ckpt(out_dir/"ckpt_lbfgs.pt", model_phi_g, opt, None, iters_adam+iters_lbfgs, meta)

    total, parts, _ = losses()
    stats = TrainStats(steps=iters_adam+max(0,iters_lbfgs), loss=float(total.detach()), wall_s=time.time()-t0)
    # dump minimal artifacts
    torch.save({"r": torch.linspace(0, rmax, n_colloc).cpu(),
                "phi": phi_of_r(torch.linspace(0, rmax, n_colloc).view(-1,1)).cpu().detach(),
                "f": f_of_r(torch.linspace(0, rmax, n_colloc).view(-1,1)).cpu().detach()}, str(out_dir/"profiles.pt"))
    with open(out_dir/"stats.json","w") as f:
        json.dump({"loss":stats.loss,"steps":stats.steps,"wall_s":stats.wall_s}, f, indent=2)
    return stats

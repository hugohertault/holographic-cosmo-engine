import torch, os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from src.ekg.pinn_solver import EKGConfig, PINNScalar, train_ekg

def main():
    # Match EKGConfig signature: (L, m2L2, lamL2, r_max, n_coll, lr, steps)
    cfg = EKGConfig(L=1.0, m2L2=-2.5, lamL2=0.0, r_max=8.0, n_coll=512, lr=2e-3, steps=400)

    torch.manual_seed(1234)
    model = PINNScalar(phi0=0.5, layers=3, width=64, act="tanh")

    stats = train_ekg(cfg, model)

    # Evaluate and plot phi(r)
    device = next(model.parameters()).device
    with torch.no_grad():
        r = torch.linspace(0.0, cfg.r_max, 1024, device=device).view(-1,1)
        phi = model(r).cpu().numpy().squeeze()
    rr = r.cpu().numpy().squeeze()

    outdir = Path("enhanced_runs/quick_smoke")
    outdir.mkdir(parents=True, exist_ok=True)
    figpath = outdir / "phi.png"

    plt.figure(figsize=(6,3.5))
    plt.plot(rr, phi, lw=2)
    plt.xlabel("r")
    plt.ylabel("phi(r)")
    plt.title("PINN scalar — quick smoke")
    plt.tight_layout()
    plt.savefig(figpath, dpi=150)
    plt.close()

    print("Run stats:", stats)
    print("Saved figure:", figpath)

if __name__ == "__main__":
    main()

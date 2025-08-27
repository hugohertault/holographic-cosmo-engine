import os, json, numpy as np, matplotlib.pyplot as plt, pathlib
import torch
from src.ekg.pinn_solver_ekg import EKGConfig, PINNScalarEKG, train_ekg

def main():
    out_dir = pathlib.Path("enhanced_runs/quick_smoke_ekg")
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = EKGConfig(
        n_colloc=512,
        L=1.0,
        m2L2=-2.5,   # BF-bound friendly
        lamL2=0.0,
        r_max=8.0,
        lr=2e-3,
        iters_adam=400,
        seed=1234,
        w_tail=1.0,
        w_center=0.1,
        w_smooth=0.0,
        w_DEC=0.0
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PINNScalarEKG(hidden=64, layers=4)

    stats = train_ekg(cfg, model, device=device)
    print("Run stats:", stats)

    # Save fields on a fine grid
    with torch.no_grad():
        r = torch.linspace(0.0, cfg.r_max, 600).view(-1,1)
        phi = model(r).cpu().numpy().reshape(-1)
        r_np = r.cpu().numpy().reshape(-1)
        # AdS reference metric for plotting
        f = 1.0 + (r_np/cfg.L)**2
        N = np.sqrt(f)
        A = 1.0/N

    np.save(out_dir/"r.npy", r_np)
    np.save(out_dir/"phi.npy", phi)
    np.save(out_dir/"N.npy", N)
    np.save(out_dir/"A.npy", A)

    # Plot phi and metric
    plt.figure(figsize=(6,3.4))
    plt.plot(r_np, phi, label="phi(r)")
    plt.xlabel("r"); plt.ylabel("phi")
    plt.title("PINN scalar on AdS5 (smoke)")
    plt.tight_layout()
    plt.savefig(out_dir/"phi.png", dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(6,3.4))
    plt.plot(r_np, N, label="N(r)")
    plt.plot(r_np, A, label="A(r)")
    plt.xlabel("r"); plt.ylabel("metric functions (AdS ref)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir/"metric.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("Saved:", out_dir/"phi.png", "and", out_dir/"metric.png")

if __name__ == "__main__":
    main()

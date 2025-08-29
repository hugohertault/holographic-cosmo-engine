import argparse, pathlib, json, math, os
from .engine import train_hybrid

def main_train():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=False)
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--iters_adam", type=int, default=800)
    p.add_argument("--iters_lbfgs", type=int, default=200)
    p.add_argument("--adam_lr", type=float, default=2e-3)
    p.add_argument("--use_amp", action="store_true")
    args = p.parse_args()

    # minimal inline defaults (config can override later)
    cfg = dict(m2L2=-2.5, lamL2=0.5, L=1.0, n_colloc=512, rmax=8.0, w_kg=1.0, w_ein=0.01, w_tail=0.1, w_center=0.0)
    if args.config and pathlib.Path(args.config).exists():
        with open(args.config) as f: cfg.update(json.load(f) if args.config.endswith(".json") else {})

    stats = train_hybrid(
        out_dir=pathlib.Path(args.out),
        m2L2=cfg["m2L2"], lamL2=cfg["lamL2"], L=cfg["L"],
        n_colloc=cfg["n_colloc"], rmax=cfg["rmax"],
        iters_adam=args.iters_adam, iters_lbfgs=args.iters_lbfgs, adam_lr=args.adam_lr,
        w_kg=cfg["w_kg"], w_ein=cfg["w_ein"], w_tail=cfg["w_tail"], w_center=cfg["w_center"],
        use_amp=args.use_amp,
        resume=(pathlib.Path(args.resume) if args.resume else None),
    )
    print(f"Done. steps={stats.steps} loss={stats.loss:.3e} wall={stats.wall_s:.1f}s")

def main_analyze():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=str, required=True)
    p.add_argument("--Delta_plus", type=float, required=False, default=3.224744871391589)
    args = p.parse_args()
    run = pathlib.Path(args.run)
    prof = torch.load(str(pathlib.Path(run)/"profiles.pt"), map_location="cpu")
    r = prof["r"].numpy().ravel(); phi = prof["phi"].numpy().ravel()
    # lightweight check: effective falloff slope
    import numpy as np, matplotlib.pyplot as plt
    eps = 1e-9
    mask = r > (0.7*r.max())
    reff = r[mask]; phit = np.abs(phi[mask]) + eps
    y = -np.gradient(np.log(phit), np.log(reff+eps))
    print({"Delta_eff_tail_mean": float(y.mean()), "Delta_plus_theory": args.Delta_plus})
    # quick plot
    fig = plt.figure()
    plt.loglog(r+eps, np.abs(phi)+eps)
    plt.title("phi(r) (abs)"); plt.xlabel("r"); plt.ylabel("|phi|")
    out = pathlib.Path(run)/"quick_phi.png"; fig.savefig(out, dpi=120); plt.close(fig)
    print(f"Saved {out}")

if __name__ == "__main__":
    # allow: python -m hce.cli main_train ...
    import sys
    if len(sys.argv) > 1 and sys.argv[1] in ("main_train","main_analyze"):
        globals()[sys.argv[1]]()
    else:
        main_train()

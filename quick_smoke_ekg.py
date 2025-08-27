import json
from pathlib import Path
from src.ekg.pinn_solver_ekg import EKGConfig, PINNScalar, train_ekg

def main():
    outdir = Path("enhanced_runs/quick_smoke_ekg")
    cfg = EKGConfig(
        r_max=8.0,
        n_colloc=512,
        m2L2=-2.5,
        lamL2=0.0,
        L=1.0,
        w_tail=1.0,
        w_center=1e-4,
        lr=2e-3,
        iters_adam=400,
        seed=1234
    )
    model = PINNScalar(width=64, depth=4)
    stats = train_ekg(cfg, model, outdir)
    print("Run stats:", stats)
    # save small json
    with open(outdir/"stats.json","w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved: {stats['figure']}")

if __name__ == "__main__":
    main()

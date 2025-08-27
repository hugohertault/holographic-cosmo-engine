
from src.ekg.pinn_solver_ekg import EKGConfig, PINNScalar, train_ekg
from pathlib import Path

def main():
    cfg = EKGConfig(r_max=8.0, n_colloc=768, m2L2=-2.5, lamL2=0.5, L=1.0,
                    w_tail=1.0, w_center=1e-6, lr=2e-3, iters_adam=600, seed=1234,
                    learn_y=True, lbfgs_iters=150)
    model = PINNScalar(width=64, depth=4, learn_y=True)
    stats = train_ekg(cfg, model, out_dir=Path("enhanced_runs/quick_smoke_ekg_learny"))
    print("Run stats:", stats)
    print("Figure:", stats["figure"])

if __name__ == "__main__":
    main()

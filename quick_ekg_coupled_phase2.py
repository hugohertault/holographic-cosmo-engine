
import sys, os, json, math, datetime
from pathlib import Path
# ensure local src on path when run from repo root
HERE = Path(__file__).resolve().parent
SRC = HERE / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ekg.system import EKGCfg, train_ekg, delta_plus_from_m2L2

def main():
    out = HERE / "enhanced_runs" / "coupled_phase2"
    out.mkdir(parents=True, exist_ok=True)

    cfg = EKGCfg(
        rmax=8.0,
        n_colloc=512,
        L=1.0,
        m2L2=-2.5,      # inside BF window
        lamL2=0.3,      # small quartic to stabilize
        iters_adam=800,
        iters_lbfgs=150,
        lr=2e-3,
        w_kg=1.0,
        w_tail=5.0,
        w_center=1e-3,
        w_ein=0.2,      # Einstein on
        w_nec=0.1,
        w_dec=0.1,
        delta_plus=None,  # auto from m2L2
        tail_frac=0.10,
        device="cuda" if False and __import__("torch").cuda.is_available() else "cpu",
        seed=1234
    )
    r, phi, f, N, stats = train_ekg(cfg, out_dir=out)
    print("=== DONE ===")
    print("Stats:", json.dumps(stats, indent=2))
    print("Saved dir:", out)

if __name__ == "__main__":
    main()

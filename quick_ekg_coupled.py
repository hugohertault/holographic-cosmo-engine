# quick_ekg_coupled.py
import sys, os
from pathlib import Path
# expose local src/ to imports
sys.path.insert(0, str(Path(__file__).resolve().parents[0] / "src"))

from ekg.system import EKGCfg, train_ekg
import json

def main():
    out = Path("enhanced_runs/coupled_demo")
    out.mkdir(parents=True, exist_ok=True)
    cfg = EKGCfg(
        rmax=8.0, n_colloc=512, m2L2=-2.5, lamL2=0.5, L=1.0,
        w_kg=1.0, w_tail=1.0, w_center=1.0,
        iters_adam=600, lr=5e-4, device="cpu", seed=1234
    )
    r, phi, f, N, stats = train_ekg(cfg, out_dir=out)
    meta = {"Delta_plus": stats.get("Delta_plus", None)}
    (out/"meta.json").write_text(json.dumps(meta, indent=2))
    print("=== DONE === Saved ->", out)

if __name__ == "__main__":
    main()

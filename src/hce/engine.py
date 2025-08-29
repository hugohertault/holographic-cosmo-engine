from pathlib import Path
from typing import Optional, Dict, Any
import json
from .config import TrainConfig

# Reuse your existing solver
from ekg.system import EKGCfg, train_ekg

def run_train(cfg: TrainConfig, out_dir: str | Path) -> Dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Map TrainConfig -> EKGCfg (same field names as in your current system.py)
    ekg_cfg = EKGCfg(
        rmax=cfg.rmax,
        n_colloc=cfg.n_colloc,
        m2L2=cfg.m2L2,
        lamL2=cfg.lamL2,
        L=cfg.L,
        w_kg=cfg.w_kg,
        w_tail=cfg.w_tail,
        w_center=cfg.w_center,
        w_ein=cfg.w_ein,
        w_nec=cfg.w_nec,
        w_dec=cfg.w_dec,
        iters_adam=cfg.iters_adam,
        lr=cfg.lr,
        device=cfg.device,
    )
    r, phi, f, N, stats = train_ekg(ekg_cfg, out_dir=out)
    # Save a lightweight JSON
    (out/"hce_summary.json").write_text(json.dumps(stats, indent=2))
    return stats

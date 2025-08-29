import argparse, sys
from pathlib import Path
from .config import TrainConfig
from .engine import run_train

def _add_common(o):
    o.add_argument("--config", type=str, help="YAML TrainConfig", required=True)
    o.add_argument("--out", type=str, help="output dir", required=True)

def main_train(argv=None):
    p = argparse.ArgumentParser(prog="hce-train")
    _add_common(p)
    args = p.parse_args(argv)
    cfg = TrainConfig.from_yaml(args.config)
    stats = run_train(cfg, args.out)
    print("=== hce-train done ===")
    print(stats)

def main_analyze(argv=None):
    from analysis.ekg_analysis import run_analysis
    p = argparse.ArgumentParser(prog="hce-analyze")
    p.add_argument("--run", type=str, required=True, help="run dir with r.txt/phi.txt/f.txt/N.txt")
    p.add_argument("--Delta_plus", type=float, required=True)
    args = p.parse_args(argv)
    summary = run_analysis(Path(args.run), Delta_plus=args.Delta_plus)
    print("=== hce-analyze summary ===")
    print(summary)

def main_spectrum(argv=None):
    from spectrum_sparse import eigs_sparse
    import numpy as np
    p = argparse.ArgumentParser(prog="hce-spectrum")
    p.add_argument("--run", type=str, required=True)
    p.add_argument("--k", type=int, default=6)
    p.add_argument("--which", type=str, default="SA")
    args = p.parse_args(argv)
    run = Path(args.run)
    r = np.loadtxt(run/"r.txt")
    phi = np.loadtxt(run/"phi.txt")
    vals, vecs, _ = eigs_sparse(r, phi, m2L2=-2.5, lamL2=0.5, L=1.0, k=args.k, which=args.which)
    print("eigs:", vals[:args.k])

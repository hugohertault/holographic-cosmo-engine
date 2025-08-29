
import os, json, math, argparse
from pathlib import Path
from analysis.ekg_analysis import run_analysis

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="enhanced_runs/analysis_ekg")
    ap.add_argument("--m2L2", type=float, default=-2.5)
    ap.add_argument("--lamL2", type=float, default=0.5)
    ap.add_argument("--L", type=float, default=1.0)
    ap.add_argument("--Delta_plus", type=float, default=float("nan"))
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[0]
    Delta_plus = None if (args.Delta_plus!=args.Delta_plus) else args.Delta_plus
    out_dir = repo/args.out
    summary = run_analysis(repo_root=repo, out_dir=out_dir, m2L2=args.m2L2, lamL2=args.lamL2, L=args.L, Delta_plus=Delta_plus)
    print("=== Analysis summary ===")
    print(json.dumps(summary, indent=2))

if __name__=="__main__":
    main()

from scipy import integrate
import sys, json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

import sys, json
from pathlib import Path
HERE = Path(__file__).resolve().parent
SRC = HERE / 'src'
if str(SRC) not in sys.path: sys.path.insert(0, str(SRC))
from spectrum_sparse import eigs_sparse
# Reuse demo profile if available:
try:
    from spectrum import demo_profile
except Exception:
    demo_profile = None

REPO = Path(__file__).resolve().parents[0]
OUT  = REPO / "enhanced_runs" / "sparse_demo"
OUT.mkdir(parents=True, exist_ok=True)

# Load r,phi if present; else demo
run_demo = True
rfile = REPO/"enhanced_runs"/"notebook_test"/"r.npy"
pfile = REPO/"enhanced_runs"/"notebook_test"/"phi.npy"
if rfile.exists() and pfile.exists():
    r = np.load(rfile); phi = np.load(pfile)
    m2L2=-2.5; L=1.0; lamL2=0.5
    Delta_plus = 2.0 + np.sqrt(4.0 + m2L2)
    run_demo = False
elif demo_profile is not None:
    r, phi, Delta_plus = demo_profile(rmax=8.0, N=800, m2L2=-2.5)
    m2L2=-2.5; L=1.0; lamL2=0.5
else:
    # Fallback simple profile
    r = np.linspace(0, 8.0, 800)
    phi = 0.5*np.exp(-r)
    m2L2=-2.5; L=1.0; lamL2=0.5
    Delta_plus = 2.0 + np.sqrt(4.0 + m2L2)

vals, vecs, U = eigs_sparse(r, phi, m2L2=m2L2, lamL2=lamL2, L=L, Delta_plus=Delta_plus, k=8, which="SA")
print("First eigenvalues (sparse):", vals[:8])

# Save quick plot
plt.figure(figsize=(6,3))
for j in range(min(4, vecs.shape[1])):
    y = vecs[:,j]
    y = y/np.sqrt(integrate.trapezoid(y*y, r))  # normalize by L2 on r
    plt.plot(r, y + 0.5*j, lw=1)
plt.title("First few eigenmodes (offset)")
plt.xlabel("r"); plt.ylabel("y_n (offset)")
plt.tight_layout()
png = OUT/"sparse_modes.png"
plt.savefig(png, dpi=130, bbox_inches="tight")
print("Saved:", png)

meta = {
    "eigs": vals[:8].tolist(),
    "m2L2": m2L2, "lamL2": lamL2, "Delta_plus": float(Delta_plus),
    "used_demo": run_demo
}
with open(OUT/"sparse_meta.json","w") as f: json.dump(meta, f, indent=2)
print("Saved:", OUT/"sparse_meta.json")


"""
Portable spectrum module for holographic-cosmo-engine.

- Builds a 1D Schrödinger-like operator L y = -y'' + U(r) on a non-uniform grid.
- Boundary conditions:
    * Neumann at r=0: y'(0)=0 (mirror ghost, consistent second derivative)
    * Robin at r=rmax: y'(rmax) + alpha * y(rmax) = 0 (one-sided)
- Computes a few lowest eigenvalues/eigenvectors using scipy.sparse.linalg.eigsh
- CLI entry-point works in scripts and notebooks (no reliance on __file__).
"""

import numpy as np
from pathlib import Path

try:
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
except Exception as e:
    raise RuntimeError("scipy is required for spectrum.py. Please install scipy.") from e


def _tridiag_nonuniform_second_derivative(r):
    """
    Build the tridiagonal coefficients (a,b,c) that approximate -d^2/dr^2
    on a non-uniform grid r[0..N-1], with:
      - Neumann at i=0: y'(0)=0 (mirror ghost => y''(0) ≈ 2*(y1 - y0)/h0^2)
      - interior i: standard non-uniform second derivative (conservative form)
      - last i=N-1: handle separately via Robin condition later (we return a,b,c
        with last row for pure second derivative on uniform last spacing; the BC
        will be merged in the operator builder)
    Returns:
      a (N,): lower diagonal
      b (N,): main diagonal
      c (N,): upper diagonal
    """
    r = np.asarray(r, dtype=float)
    N = r.size
    if N < 3:
        raise ValueError("Need at least N>=3 points for a meaningful second derivative.")

    a = np.zeros(N, dtype=float)
    b = np.zeros(N, dtype=float)
    c = np.zeros(N, dtype=float)

    # spacing
    h = np.diff(r)  # size N-1, h[i] = r[i+1]-r[i]
    if np.any(h <= 0):
        raise ValueError("r must be strictly increasing.")

    # i=0 (Neumann y'(0)=0): y''(0) ≈ 2*(y1 - y0)/h0^2
    h0 = h[0]
    b[0] =  2.0 / (h0*h0)   # multiplies y0
    c[0] = -2.0 / (h0*h0)   # multiplies y1
    # a[0] stays 0

    # interior 1..N-2, standard nonuniform formula
    for i in range(1, N-1):
        hm = r[i] - r[i-1]  # h_{i-1}
        hp = r[i+1] - r[i]  # h_i
        denom = (hm + hp)
        # -y'': a*y_{i-1} + b*y_i + c*y_{i+1}
        a[i] = -2.0 / (hm * denom)
        c[i] = -2.0 / (hp * denom)
        b[i] =  2.0 / (hm * hp)

    # i=N-1: we leave placeholder for pure second deriv with last spacing,
    # BC will be injected in operator builder (Robin)
    # for a consistent second-derivative, assume local uniform h_last = h[N-2]
    h_last = h[-1]
    a[N-1] = -1.0 / (h_last*h_last)
    b[N-1] =  2.0 / (h_last*h_last)
    c[N-1] = -1.0 / (h_last*h_last)

    return a, b, c


def build_operator(r, U, alpha_robin=0.0):
    """
    Return sparse CSR matrix A for L y = -y'' + U(r) y, with:
      - Neumann at r=0
      - Robin at rmax: y'(rmax) + alpha_robin * y(rmax) = 0
    Args:
      r : (N,) strictly increasing grid
      U : (N,) potential values
      alpha_robin : float, Robin coefficient at rmax
    """
    r = np.asarray(r, dtype=float)
    U = np.asarray(U, dtype=float)
    if r.shape != U.shape:
        raise ValueError("r and U must have the same shape")

    N = r.size
    a, b, c = _tridiag_nonuniform_second_derivative(r)

    # Impose Robin at last node using a ghost point elimination:
    # y'(N-1) ≈ (y_N - y_{N-2}) / (2 h_{N-2})  => y_N = y_{N-2} - 2 h a y_{N-1}
    # y''(N-1) ≈ (y_N - 2 y_{N-1} + y_{N-2}) / h^2
    # Substitute y_N to get row coefficients:
    h = r[-1] - r[-2]
    a_last = -2.0 / (h*h)
    b_last = (2.0 + 2.0 * alpha_robin * h) / (h*h)
    # overwrite last row second-derivative stencil:
    a[-1], b[-1], c[-1] = a_last, b_last, 0.0

    # Add potential: b += U
    b = b + U

    # Sparse assemble
    diags = []
    offsets = []
    # lower
    diags.append(a[1:]); offsets.append(-1)
    # main
    diags.append(b);     offsets.append(0)
    # upper
    diags.append(c[:-1]); offsets.append(1)
    A = sp.diags(diagonals=diags, offsets=offsets, shape=(N, N), format="csr")
    return A


def U_from_phi(r, phi, m2L2=-2.5, lamL2=0.0, L=1.0):
    """
    Build an effective potential U(r).
    Minimal model: mass term around background scalar profile phi(r):
      V(phi) = (m^2/2) phi^2 + (lambda/4) phi^4
      U_eff ~ V''(phi) = m^2 + 3 lambda phi^2
    Here m^2 = m2L2 / L^2, lambda = lamL2 / L^2.
    """
    r = np.asarray(r, dtype=float)
    phi = np.asarray(phi, dtype=float)
    if r.shape != phi.shape:
        raise ValueError("r and phi must have same shape")

    m2 = m2L2 / (L*L)
    lam = lamL2 / (L*L)
    U = m2 + 3.0 * lam * (phi**2)
    # small radial "centrifugal" stabilization if desired (optional):
    # add_eps = 1e-6 / (1.0 + r*r)
    # U = U + add_eps
    return U


def compute_spectrum(r, phi, m2L2=-2.5, lamL2=0.0, L=1.0, Delta_plus=None, k=6):
    """
    Compute k lowest eigenvalues of L = -d2/dr2 + U(r) with Neumann@0, Robin@rmax.
    Robin alpha is set from the AdS falloff exponent Delta_plus (if provided):
      alpha_robin = (Delta_plus - 4) / rmax
    Args:
      r, phi : arrays
      m2L2, lamL2, L : potential parameters
      Delta_plus : if None, no Robin shift (alpha=0). Otherwise alpha=(Dp-4)/rmax
      k : number of eigenpairs
    Returns:
      evals (k,), evecs (N,k)
    """
    r = np.asarray(r, dtype=float)
    phi = np.asarray(phi, dtype=float)
    if r.ndim != 1:
        raise ValueError("r must be 1D.")
    if r.shape != phi.shape:
        raise ValueError("r and phi must have the same shape.")

    # Potential
    U = U_from_phi(r, phi, m2L2=m2L2, lamL2=lamL2, L=L)

    # Robin alpha from Delta_plus
    if Delta_plus is None:
        alpha = 0.0
    else:
        rmax = r[-1]
        alpha = (float(Delta_plus) - 4.0) / max(rmax, 1e-8)

    # Operator
    A = build_operator(r, U, alpha_robin=alpha)

    # Use sparse symmetric eigensolver
    # Matrix is real symmetric; ask for smallest algebraic
    try:
        evals, evecs = spla.eigsh(A, k=min(k, A.shape[0]-2), which="SA")
        idx = np.argsort(evals)
        evals = np.array(evals)[idx]
        evecs = np.array(evecs)[:, idx]
    except Exception as e:
        # Fallback to dense for tiny N
        A_dense = A.toarray()
        evals, evecs = np.linalg.eigh(A_dense)
        idx = np.argsort(evals)
        evals = evals[idx][:k]
        evecs = evecs[:, idx][:, :k]

    return evals, evecs


def _find_latest_run(repo_root):
    """
    Try to locate a run directory containing r.npy and phi.npy.
    Search under: enhanced_runs/*/
    Returns Path or None.
    """
    root = Path(repo_root)
    cand = []
    base = root / "enhanced_runs"
    if base.exists():
        for d in base.glob("*"):
            if not d.is_dir(): 
                continue
            if (d/"r.npy").exists() and (d/"phi.npy").exists():
                cand.append(d)
    if not cand:
        return None
    # latest by mtime
    cand.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return cand[0]


def _demo_profile(N=300, rmax=8.0):
    """
    Build a monotonic grid and a toy phi profile that decays with r.
    """
    r = np.linspace(0.0, rmax, N)
    phi = 0.8 * np.exp(-r) * (1.0 + 0.2*np.cos(2.0*r))
    return r, phi


def main_cli(repo_root=".", target_run=None, k=8):
    """
    CLI entry that tries to load (r.npy, phi.npy) from latest run, or builds a demo.
    Prints first k eigenvalues and writes spectrum.png if matplotlib is available.
    """
    repo_root = Path(repo_root)
    run_dir = None
    if target_run is not None:
        d = repo_root / target_run
        if (d/"r.npy").exists() and (d/"phi.npy").exists():
            run_dir = d
    if run_dir is None:
        latest = _find_latest_run(repo_root)
        if latest is not None:
            run_dir = latest

    if run_dir is not None:
        r = np.load(run_dir/"r.npy")
        phi = np.load(run_dir/"phi.npy")
        print(f"Using run: {run_dir}")
    else:
        r, phi = _demo_profile()
        print("No run found → using demo profile.")

    # Heuristic Delta_plus from m2L2
    m2L2 = -2.5
    Delta_plus = 2.0 + np.sqrt(4.0 + m2L2)
    evals, evecs = compute_spectrum(r, phi, m2L2=m2L2, lamL2=0.5, L=1.0,
                                    Delta_plus=Delta_plus, k=k)
    print("First eigenvalues:")
    print(evals)

    # Optional plot
    try:
        import matplotlib.pyplot as plt
        import os
        out_dir = Path(repo_root) / "enhanced_runs" / "spectrum"
        out_dir.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(5,3))
        for j in range(min(k, evecs.shape[1])):
            y = evecs[:, j]
            y = y/np.max(np.abs(y)+1e-12)
            plt.plot(r, y, label=f"mode {j} ({evals[j]:.3g})")
        plt.xlabel("r"); plt.ylabel("normalized eigenmodes")
        plt.legend(fontsize=8, ncol=2)
        plt.tight_layout()
        figpath = out_dir / "spectrum.png"
        plt.savefig(figpath, dpi=150, bbox_inches="tight")
        print("Saved:", figpath)
    except Exception as e:
        print("Plot skipped:", repr(e))


if __name__ == "__main__":
    # Notebook-safe: __file__ may not exist; default to current working dir.
    repo = Path.cwd()
    try:
        main_cli(repo_root=repo, target_run=None, k=8)
    except Exception as e:
        print("Spectrum CLI error:", repr(e))


def save_spectrum_plot(r, evals, out_png, title="Spectrum"):
    import numpy as np, matplotlib.pyplot as plt, os
    plt.figure(figsize=(6,3.2))
    idx = np.arange(len(evals))
    plt.scatter(idx, evals)
    plt.plot(idx, evals)
    plt.axhline(0.0, linestyle="--", linewidth=1)
    plt.xlabel("Mode index")
    plt.ylabel("Eigenvalue lambda")
    plt.title(title)
    plt.tight_layout()
    os.makedirs(os.path.dirname(str(out_png)), exist_ok=True)
    plt.savefig(str(out_png), dpi=150, bbox_inches="tight")
    plt.close()


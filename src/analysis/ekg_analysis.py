
import json, math
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def _glob_runs(repo_root: Path):
    root = repo_root/"enhanced_runs"
    if not root.exists(): return []
    cands = []
    for sub in root.iterdir():
        if not sub.is_dir(): continue
        # on prend les runs couplés phase2 par défaut
        if sub.name.startswith("coupled_phase2"):
            cands.append(sub)
    return sorted(cands, key=lambda p: p.stat().st_mtime, reverse=True)

def load_latest_run(repo_root: Path):
    """Charge le dernier run couplé: retourne dict {'r','phi','f','N','meta','dir'}.
       Essaie .npy ; si absent, essaie .json meta pour indices.
    """
    runs = _glob_runs(repo_root)
    if not runs:
        raise FileNotFoundError("Aucun dossier enhanced_runs/coupled_phase2* trouvé.")
    run = runs[0]
    r = np.load(run/"r.npy") if (run/"r.npy").exists() else None
    phi = np.load(run/"phi.npy") if (run/"phi.npy").exists() else None
    f = np.load(run/"f.npy") if (run/"f.npy").exists() else None
    N = np.load(run/"N.npy") if (run/"N.npy").exists() else None
    meta = {}
    if (run/"meta.json").exists():
        with open(run/"meta.json") as fmeta: meta = json.load(fmeta)
    # tolère que f/N manquent; mais r,phi sont indispensables pour l’analyse minimale
    if r is None or phi is None:
        raise FileNotFoundError(f"r.npy/phi.npy manquants dans {run}. Relancer quick_ekg_coupled_phase2.py pour sauver les arrays.")
    return {"dir":run, "r":r.squeeze(), "phi":phi.squeeze(), "f": (None if f is None else f.squeeze()), "N": (None if N is None else N.squeeze()), "meta":meta}

def finite_diff(y, x):
    """d y / d x (non-uniform) simple et robuste."""
    y = np.asarray(y, float); x = np.asarray(x, float)
    n = len(x)
    dy = np.zeros_like(y)
    # central
    for i in range(1,n-1):
        dxp = x[i+1]-x[i]
        dxm = x[i]-x[i-1]
        dy[i] = (dxm*(y[i+1]-y[i]) + dxp*(y[i]-y[i-1]))/(dxp*dxm*(dxp+dxm))*(dxp*dxm*(dxp+dxm)) # placeholder algebra
    # plus simple et stable:
    for i in range(1,n-1):
        dy[i] = (y[i+1]-y[i-1])/(x[i+1]-x[i-1])
    dy[0]  = (y[1]-y[0])/(x[1]-x[0])
    dy[-1] = (y[-1]-y[-2])/(x[-1]-x[-2])
    return dy

def effective_power_phi(r, phi, eps=1e-12):
    """Δ_eff(r) ~ - d ln|phi| / d ln r (utile près du bord si phi ~ r^{-Δ+})."""
    r = np.asarray(r, float); phi = np.asarray(phi, float)
    mask = (np.abs(phi)>eps) & (r>0)
    re = r[mask]; pe = phi[mask]
    dlogphi = finite_diff(np.log(np.abs(pe)), np.log(re))
    rp = np.zeros_like(r); rp[mask]= -dlogphi
    rp[~mask]=np.nan
    return rp

def tail_power_fit(r, phi, tail_frac=0.15):
    """Fit log|phi| ≈ c - Δ * log r sur la queue (derniers tail_frac)."""
    r = np.asarray(r, float); phi = np.asarray(phi, float)
    n = len(r); m = max(8, int(n*tail_frac))
    idx = np.arange(n-m, n)
    r_tail = r[idx]; phi_tail = np.abs(phi[idx])
    mask = (r_tail>0) & (phi_tail>0)
    x = np.log(r_tail[mask]); y = np.log(phi_tail[mask])
    if len(x)<3: return np.nan, (np.nan,np.nan)
    A = np.vstack([np.ones_like(x), -x]).T
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    c, Delta = beta[0], beta[1]
    return float(Delta), (float(c), float(np.exp(c)))

def kg_residual_ads_fixed(r, phi, m2L2=-2.5, lamL2=0.5, L=1.0):
    """Résiduel KG simplifié en AdS5 fixe: φ'' + (4/r)φ' - (m2/L^2)φ - λ φ^3 = 0
       (diagnostic brut; l’EKG complet inclut f,N).
    """
    r = np.asarray(r, float); phi = np.asarray(phi, float)
    dphi = finite_diff(phi, r)
    ddphi = finite_diff(dphi, r)
    with np.errstate(divide='ignore', invalid='ignore'):
        term = ddphi + 4.0*np.where(r>0, dphi/np.where(r==0, np.inf, r), 0.0) - (m2L2/(L*L))*phi - lamL2*(phi**3)
    term[~np.isfinite(term)] = 0.0
    return term

def run_analysis(repo_root: Path, out_dir: Path = None, m2L2=-2.5, lamL2=0.5, L=1.0, Delta_plus=None):
    data = load_latest_run(repo_root)
    r, phi, f, N = data["r"], data["phi"], data["f"], data["N"]
    meta = data["meta"]
    if Delta_plus is None:
        # si meta contient Delta_plus
        Delta_plus = meta.get("Delta_plus", 2+math.sqrt(max(0.0,4.0-m2L2)))
    if out_dir is None:
        out_dir = repo_root/"enhanced_runs"/"analysis_ekg"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) profils
    fig1 = out_dir/"profiles.png"
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6.0,4.0))
    plt.plot(r, phi, label="phi(r)")
    if f is not None: plt.plot(r, f, label="f(r)")
    if N is not None: plt.plot(r, N, label="N(r)")
    plt.xlabel("r"); plt.ylabel("fields"); plt.title("Profiles (latest run)")
    plt.legend(); plt.tight_layout(); plt.savefig(fig1, dpi=130); plt.close()

    # 2) exposant effectif
    fig2 = out_dir/"delta_eff.png"
    Delta_eff = effective_power_phi(r, phi)
    plt.figure(figsize=(6.0,3.2))
    plt.plot(r, Delta_eff, lw=1.6)
    plt.axhline(Delta_plus, ls="--", label="Δ+ (théorie)")
    plt.xlabel("r"); plt.ylabel("Δ_eff(r)")
    plt.legend(); plt.tight_layout(); plt.savefig(fig2, dpi=130); plt.close()

    # 3) fit queue
    Delta_fit, (c_fit, A_fit) = tail_power_fit(r, phi)
    fig3 = out_dir/"tail_fit.png"
    # scatter log-log + droite ajustée
    mask = (r>0) & (np.abs(phi)>0)
    x = np.log(r[mask]); y = np.log(np.abs(phi[mask]))
    xp = np.linspace(x.max()-1.0, x.max(), 100)
    yp = c_fit - Delta_fit*xp if np.isfinite(Delta_fit) else np.full_like(xp, np.nan)
    plt.figure(figsize=(6.0,3.2))
    plt.plot(x, y, '.', ms=3, alpha=0.5, label="log|phi|")
    if np.isfinite(Delta_fit):
        plt.plot(xp, yp, '-', lw=2, label=f"fit tail: Δ≈{Delta_fit:.3f}")
    plt.xlabel("log r"); plt.ylabel("log|phi|")
    plt.legend(); plt.tight_layout(); plt.savefig(fig3, dpi=130); plt.close()

    # 4) résiduel KG fixe (diagnostic)
    fig4 = out_dir/"kg_residual_ads.png"
    R = kg_residual_ads_fixed(r, phi, m2L2=m2L2, lamL2=lamL2, L=L)
    plt.figure(figsize=(6.0,3.2))
    plt.plot(r, R, lw=1.5)
    plt.xlabel("r"); plt.ylabel("R_KG(AdS fixe)")
    plt.title("Diagnostic résiduel KG (AdS5 fixe)")
    plt.tight_layout(); plt.savefig(fig4, dpi=130); plt.close()

    # 5) summary json
    summary = {
        "input_dir": str(data["dir"]),
        "Delta_plus_theory": float(Delta_plus),
        "Delta_fit_tail": float(Delta_fit) if np.isfinite(Delta_fit) else None,
        "A_fit": float(A_fit) if np.isfinite(Delta_fit) else None,
        "figures": {
            "profiles": str(fig1),
            "delta_eff": str(fig2),
            "tail_fit": str(fig3),
            "kg_residual_ads": str(fig4),
        }
    }
    with open(out_dir/"analysis_summary.json","w") as f:
        json.dump(summary, f, indent=2)
    return summary

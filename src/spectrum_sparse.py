import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh

def robin_bc_weights(Delta_plus, L=1.0):
    # Robin: y' - Delta_plus * y = 0 at r = rmax
    # Discretize with one-sided diff: y' ~ (yN - y_{N-1})/dr
    # Row enforces: (yN - yNm1)/dr - Delta_plus * yN = 0
    # -> (-1/dr) y_{N-1} + (1/dr - Delta_plus) y_N = 0
    # Return tuple for last two cols scaling; we'll plug proper dr at assembly.
    return Delta_plus

def assemble_csr_nonuniform(r, U, Delta_plus, L=1.0):
    """
    Assemble 1D operator L y = -y'' + U(r) y on non-uniform grid r[0..N-1].
    Interior: 3-pt second derivative on non-uniform mesh.
    Left BC: Neumann y'(r0)=0  -> (y1 - y0)/dr0 = 0.
    Right BC: Robin y'(rN)-Delta_plus yN=0  -> (yN - y_{N-1})/drN - Delta_plus*yN=0.
    Returns CSR matrix A (NxN).
    """
    r = np.asarray(r, float)
    U = np.asarray(U, float)
    N = r.size
    assert U.size == N and N >= 3

    rows, cols, data = [], [], []

    # helper to add entry
    def add(i,j,val):
        rows.append(i); cols.append(j); data.append(val)

    # convenience spacings
    dr = np.empty(N)
    dr[:-1] = np.diff(r)
    dr[-1]  = dr[-2]  # placeholder for form; not used in interior formula

    # Left BC: Neumann at i=0  -> (y1 - y0)/dr0 = 0
    i = 0
    dr0 = r[1]-r[0]
    add(i, 0, -1.0/dr0)
    add(i, 1,  1.0/dr0)

    # Interior 1..N-2
    for i in range(1, N-1):
        r_im1, r_i, r_ip1 = r[i-1], r[i], r[i+1]
        h_im1 = r_i - r_im1
        h_i   = r_ip1 - r_i
        # nonuniform 2nd-deriv weights (classical 3-point formula)
        a =  2.0/(h_im1*(h_im1 + h_i))
        b = -2.0/(h_im1*h_i)
        c =  2.0/(h_i*(h_im1 + h_i))
        # Operator is -y'' + U y
        add(i, i-1, -a)
        add(i, i  , -b + U[i])
        add(i, i+1, -c)

    # Right BC: Robin at i=N-1
    i = N-1
    dRN = r[-1]-r[-2]
    Dp = float(Delta_plus)
    # (yN - y_{N-1})/dRN - Dp*yN = 0
    add(i, N-2, -1.0/dRN)
    add(i, N-1,  1.0/dRN - Dp)

    A = csr_matrix((data, (rows, cols)), shape=(N, N))
    return A

def effective_potential(r, phi, m2L2=-2.5, lamL2=0.5, L=1.0):
    """
    Sample "toy" effective potential U(r) for fluctuations: m_eff^2 + lambda*phi^2.
    You can replace by the true linearized EKG potential later.
    """
    return (m2L2/(L**2)) + (lamL2/(L**2))*phi**2

def eigs_sparse(r, phi, m2L2=-2.5, lamL2=0.5, L=1.0, Delta_plus=None, k=8, which="SM"):
    """
    Build CSR operator and compute k lowest eigenvalues with ARPACK (eigsh).
    which="SM" -> smallest magnitude (good default if spectrum straddles 0).
    """
    r   = np.asarray(r, float)
    phi = np.asarray(phi, float)
    if Delta_plus is None:
        Delta_plus = 2.0 + np.sqrt(4.0 + m2L2)
    U = effective_potential(r, phi, m2L2=m2L2, lamL2=lamL2, L=L)
    A = assemble_csr_nonuniform(r, U, Delta_plus, L=L)
    # Shift to improve robustness if needed
    # Here no shift; user can try sigma with shift-invert via eigsh when necessary.
    vals, vecs = eigsh(A, k=min(k, len(r)-2), which=which)
    idx = np.argsort(vals)
    return vals[idx], vecs[:,idx], U



def _build_operator_nonuniform_local(r, phi, m2L2, lamL2, L, bc):
    """
    Construction minimale de A (CSR symétrique) pour -y'' + U(r) y
    Grille non-uniforme r[0..N-1]. Offsets [-1,0,1] -> tailles [N-1, N, N-1].
    Bord gauche: Neumann (~y'(0)=0) ; Bord droit: Robin y'(R) - Δ+/R * y(R) = 0.
    """
    import numpy as np
    import scipy.sparse as sp

    r   = np.asarray(r, float).ravel()
    phi = np.asarray(phi, float).ravel()
    N   = len(r)
    assert N >= 3, "Besoin d'au moins 3 points"

    # Pas non-uniformes
    h = np.diff(r)
    h = np.clip(h, 1e-12, None)

    # Coeffs tri-diag (discr. centrée avec moyenne harmonique)
    a = np.zeros(N)  # sous-diag (sera tronquée à N-1)
    b = np.zeros(N)  # diag
    c = np.zeros(N)  # sur-diag (sera tronquée à N-1)

    for i in range(1, N-1):
        hm = h[i-1]
        hp = h[i]
        wL = 2.0/(hm*(hm+hp))
        wR = 2.0/(hp*(hm+hp))
        a[i] = -wL
        c[i] = -wR
        b[i] =  wL + wR

    # Potentiel U(r)
    eps = 1e-9
    U = (m2L2/(L*L)) * np.ones(N)
    U += 3.0/np.maximum(r*r, eps)
    if lamL2 != 0.0:
        U += (lamL2/(L*L)) * (phi*phi)

    # Conditions de bord
    # Neumann approx à gauche: y'(0)=0 -> y1 - y0 = 0
    b[0] = 1.0
    c[0] = -1.0
    a[0] = 0.0

    # Robin à droite: y'(R) - Δ+/R * y(R) = 0 -> yN-1 - yN-2 - (Δ+/R)*yN-1 = 0
    Delta_plus = 2.0 + np.sqrt(max(0.0, 4.0 + m2L2))
    Rmax = max(r[-1], 1e-8)
    a[-1] = -1.0
    b[-1] =  1.0 + (Delta_plus / Rmax)
    c[-1] =  0.0

    # Ajout du potentiel sur la diag
    b += U

    # Tronquages conformes aux offsets [-1,0,1]
    a_sub = a[1:]     # longueur N-1
    c_sup = c[:-1]    # longueur N-1
    diags = [a_sub, b, c_sup]

    A = sp.diags(diags, offsets=[-1,0,1], shape=(N,N), format="csr")
    return A, U


def eigs_sparse(r, phi, m2L2=-2.5, lamL2=0.0, L=1.0, Delta_plus=None, k=8, which="SA",
                tol=1e-9, maxiter=20000, ncv=None, bc="robin",
                sigma_fallbacks=(0.0, "median")):
    """
    Solveur robuste d'autoval. pour opérateur 1D non-uniforme (CSR symétrique).
    Stratégie:
      (i)   essai direct eigsh(..., which='SA')
      (ii)  si échec: shift-invert sigma=0 (which='LM')
      (iii) si échec: shift-invert sigma=median(diag)
      (iv)  fallback dense eigh si N petit et k petit
    """
    import numpy as np
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla

    r   = np.asarray(r, dtype=float).ravel()
    phi = np.asarray(phi, dtype=float).ravel()
    N   = len(r)
    if ncv is None:
        ncv = min(max(2*k+2, 40), max(N-1, 50))

    # --- discretisation opérateur: -y'' + U(r) ---
    # On réutilise la construction existante A (CSR) et U depuis les helpers du module.
    # On suppose qu'il existe une fonction interne build_operator_nonuniform qui renvoie (A, U)
    # sinon on re-code ici rapidement en s'appuyant sur l'opérateur déjà utilisé dans le module.
    try:
        A, U = build_operator_nonuniform(r, phi, m2L2=m2L2, lamL2=lamL2, L=L, bc=bc)
    except NameError:
        # fallback: reconstruire avec une version simplifiée locale (copie du code du module principal)
        A, U = _build_operator_nonuniform_local(r, phi, m2L2, lamL2, L, bc)

    # bornes k
    k_eff = int(max(1, min(k, N-2)))

    # essai (i): direct 'SA'
    try:
        vals, vecs = spla.eigsh(A, k=k_eff, which="SA", tol=tol, maxiter=maxiter, ncv=ncv)
        return vals, vecs, U
    except Exception as e1:
        pass

    # essai (ii) et (iii): shift-invert
    diagA = A.diagonal()
    shifts = []
    for s in sigma_fallbacks:
        if s == "median":
            shifts.append(float(np.median(diagA)))
        else:
            shifts.append(float(s))

    for sigma in shifts:
        try:
            vals, vecs = spla.eigsh(A, k=k_eff, which="LM", sigma=sigma, tol=tol, maxiter=maxiter, ncv=ncv, mode="normal")
            return vals, vecs, U
        except Exception as e2:
            continue

    # essai (iv): dense si raisonnable
    if N <= 2500 and k_eff <= 16:
        import numpy.linalg as npl
        Ad = A.toarray()
        # eigh pour symétrique
        w, V = np.linalg.eigh(Ad)
        idx  = np.argsort(w)  # plus petites d'abord
        idxk = idx[:k_eff]
        return w[idxk], V[:, idxk], U

    # sinon on relance direct avec plus d'itérations comme ultime tentative
    vals, vecs = spla.eigsh(A, k=k_eff, which="SA", tol=max(tol,1e-8), maxiter=5*maxiter, ncv=min(ncv*2, N-1))
    return vals, vecs, U

# === end eigs_sparse ===

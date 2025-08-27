"""
Cosmological χ² module (Pantheon, BAO, RSD, WL, H0 priors, Planck DP).
"""
import numpy as np, pandas as pd

def chi2_placeholder(theta):
    # theta = [H0, Om, Obh2, w0, wa, sigma8, n_s, Neff, alpha, dmu]
    return np.sum(np.array(theta)**2)  # fake χ² for now

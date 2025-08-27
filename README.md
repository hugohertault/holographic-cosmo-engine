
# Holographic Cosmo Engine

🌌 Projet : construire un moteur holographique complet (AdS5/dS4 dual) avec :
- Solveur Einstein-Klein-Gordon via PINNs (`src/ekg`)
- Contraintes cosmologiques (SN Ia, BAO, RSD, WL, Planck) (`src/cosmo`)
- Passerelle holographique (AdS/CFT bridge) [à venir]

Repo : https://github.com/hugohertault/holographic-cosmo-engine

<!-- SWEEP_LEARNY_GALLERY_START -->
### Learn-y sweep (auto)
_Last update: 2025-08-27 07:32 UTC_


**m2L2_-3.0__lamL2_0.0** — m2L2=-3.0, lamL2=0.0, loss=1.673e-03, Dp=3.000  
![](enhanced_runs/sweep_learny/m2L2_-3.0__lamL2_0.0/phi.png)

**m2L2_-2.8__lamL2_0.0** — m2L2=-2.8, lamL2=0.0, loss=4.165e-03, Dp=3.095  
![](enhanced_runs/sweep_learny/m2L2_-2.8__lamL2_0.0/phi.png)

**m2L2_-2.5__lamL2_0.5** — m2L2=-2.5, lamL2=0.5, loss=2.684e-03, Dp=3.225  
![](enhanced_runs/sweep_learny/m2L2_-2.5__lamL2_0.5/phi.png)
<!-- SWEEP_LEARNY_GALLERY_END -->

<!-- HARDENED_RUN_START -->
### Hardened run (Adam + LBFGS)
_Last update: 2025-08-27 07:37 UTC_

**hardened_m2L2_-2.5_lamL2_0.5** — m2L2=-2.5, lamL2=0.5,  
loss≈1.741e-02, Δ+≈3.225  

![](enhanced_runs/hardened/hardened_m2L2_-2.5_lamL2_0.5/phi_refined.png)
<!-- HARDENED_RUN_END -->

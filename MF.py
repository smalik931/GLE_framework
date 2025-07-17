#!/usr/bin/env python3
"""
Memory-function extraction (pure-Python loops) for 3 merged VACF sets,
truncated to 50 ps.

Input file:
    vacf_T248_merged.txt    (columns: time  set1  set2  set3)

Outputs:
    MemoryFunction-248-Set1.txt … Set3.txt      (t ≤ 50 ps)
    MemoryFunction-248-Set1-LongTail.txt …      (analytic tail)
    MemoryFunction-248-Zoomed.txt               (power-law thinning)
"""
import numpy as np
from scipy.linalg import solve
import os, sys

# --------------------------- user knobs --------------------------------
VACF_FILE   = "vacf_T248_merged.txt"
MAX_TIME_PS = 50.0         # truncate at 50 ps
H_EXP       = 1.7          # exponent for zoom thinning

# Hard-coded fit parameters for the 3 sets
D_values     = np.array([0.000158019, 0.000162059, 0.000159380])
Alpha_values = np.array([0.258693966, 0.252703075, 0.258746491])
# -----------------------------------------------------------------------

if not os.path.isfile(VACF_FILE):
    sys.exit(f"VACF file '{VACF_FILE}' not found.")

data = np.loadtxt(VACF_FILE)
time = data[:, 0]
sets = data[:, 1:4]   # three VACF columns
n_sets = sets.shape[1]

# determine dt and max index for 50 ps
dt = time[1] - time[0]
max_idx = min(len(time), int(MAX_TIME_PS/dt) + 1)
P = max_idx - 1

# pre-compute zoom indices and zoom time line
t_zoom_indices = [int(i**H_EXP) for i in range(1, int(P**(1/H_EXP))) if int(i**H_EXP) < P]
zoom_t = (np.arange(P) * dt)[t_zoom_indices]

# Boltzmann prefactor (for tail)
kB = 1.380649e-23
Mass = 677.945 * 1.66054e-27
v2 = (kB * 248) / Mass * 1e18

zoom_blocks = []
# loop over sets
for s in range(n_sets):
    vacf = sets[:max_idx, s]

    # build correlation vector and A matrix\ n
    corr = vacf[:P]
    A = np.zeros((P, P))
    for j in range(P):
        for i in range(j+1):
            A[j, i] = corr[j - i]
    w = np.ones(P)
    w[0] = w[-1] = 0.5
    A *= w[:, None]

    # build b vector
    b = np.array([-(vacf[i+1] - vacf[i]) / dt**2 for i in range(P)])

    # solve for kappa
    kappa = solve(A, b)

    # save κ(t)
    t_vals = np.arange(P) * dt
    np.savetxt(
        f"MemoryFunction-248-Set{s+1}.txt",
        np.column_stack((t_vals, kappa)),
        header="t [ps]  kappa(t)", fmt="%.12f"
    )

    # analytic long-time tail
    D, alpha = D_values[s], Alpha_values[s]
    tlt = np.arange(10.0, 40.0, 0.1)
    tail = (v2 / D) * (np.sin(np.pi * alpha) / (np.pi * alpha)) * tlt**(-alpha)
    np.savetxt(
        f"MemoryFunction-248-Set{s+1}-LongTail.txt",
        np.column_stack((tlt, tail)),
        header="t [ps]  long_tail", fmt="%.12f"
    )

    # collect zoomed values
    zoom_blocks.append(kappa[t_zoom_indices])

# write zoomed file
zoom_mat = np.column_stack([zoom_t] + zoom_blocks)
header = "t [ps]  " + "  ".join(f"Set{i+1}" for i in range(n_sets))
np.savetxt(
    "MemoryFunction-248-Zoomed.txt",
    zoom_mat,
    header=header,
    fmt="%.12f"
)

print("Done up to 50 ps for 3 sets.")


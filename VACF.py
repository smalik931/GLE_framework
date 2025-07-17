#!/usr/bin/env python3
"""Memory-safe VACF/LTT analysis **with variable trajectory length support**.

Fixes the "ValueError: dimensions must match" error that appears when the
kept trajectory fragment (`KEEP_FRACTION`) contains fewer frames than
`MAX_LAG_FRAMES`.  The script now:
  • uses `n_lag = min(MAX_LAG_FRAMES, N_keep)` per set;
  • trims all sets to the *shortest* common `n_lag` before stacking;
  • thus guarantees `time_arr` and `vacf_i` have identical lengths.

Configuration is unchanged. Only NumPy/pyFFTW/CuPy are required.
"""
from __future__ import annotations

import os, time
from pathlib import Path
from typing import Dict, Tuple

import h5py
import numpy as np

# ───────────────────────────── CONFIG ──────────────────────────────
KEEP_FRACTION   = 0.05           # keep first 5 % of each trajectory
MAX_LAG_FRAMES  = 150_000        # desired VACF lags (0.01 ps → 1.5 ns)
CHUNK_FRAMES    = 262_144        # ≥ MAX_LAG_FRAMES, power-of-two
N_FFT_THREADS   = os.cpu_count() or 1
LTT_RANGE_PS    = (1, 50, 1)
BACKEND_VERBOSE = False

# ───────────────────────────── FFT backend ─────────────────────────
try:
    import pyfftw.interfaces.numpy_fft as fft
    fft.interfaces.cache.enable()
    _FFT_BACKEND = f"pyFFTW ({N_FFT_THREADS} threads)"
except ModuleNotFoundError:
    try:
        import cupy as cp
        fft = cp.fft                        # type: ignore
        _FFT_BACKEND = "CuPy (GPU)"
    except ModuleNotFoundError:
        import numpy.fft as fft             # type: ignore
        _FFT_BACKEND = "NumPy (single-thread)"

if BACKEND_VERBOSE:
    print(f"[VACF] Using {_FFT_BACKEND}\n")

# ───────────────────────────── HELPERS ────────────────────────────

def read_fit_parameters(fname: str) -> Dict[int, Tuple[float, float]]:
    raw = np.loadtxt(fname, skiprows=1, dtype=str)
    out: Dict[int, Tuple[float, float]] = {}
    for row in raw:
        try:
            out[int(row[0])] = (float(row[1]), float(row[2]))
        except ValueError:
            continue
    return out


def truncated_vacf(vx: np.ndarray, vy: np.ndarray, maxlag: int) -> np.ndarray:
    """VACF via FFT blocks, memory-safe."""
    assert maxlag <= CHUNK_FRAMES, "CHUNK_FRAMES must be ≥ maxlag"

    vx = vx.astype("float32")
    vy = vy.astype("float32")
    vx -= vx.mean(0, keepdims=True)
    vy -= vy.mean(0, keepdims=True)

    N = vx.shape[0]
    vacf = np.zeros(maxlag)
    counts = np.zeros(maxlag, dtype=int)

    kwargs = {"axis": 0}
    if _FFT_BACKEND.startswith("pyFFTW"):
        kwargs["threads"] = N_FFT_THREADS

    for start in range(0, N, CHUNK_FRAMES):
        blk_x = vx[start : min(start + CHUNK_FRAMES, N)]
        blk_y = vy[start : min(start + CHUNK_FRAMES, N)]
        pad = CHUNK_FRAMES * 2
        f_x = fft.rfft(blk_x, n=pad, **kwargs)
        f_y = fft.rfft(blk_y, n=pad, **kwargs)
        ac_x = fft.irfft(f_x.conj() * f_x, n=pad, **kwargs)[: len(blk_x)].real
        ac_y = fft.irfft(f_y.conj() * f_y, n=pad, **kwargs)[: len(blk_y)].real
        ac = 0.5 * (ac_x + ac_y).mean(1)
        n_valid = min(maxlag, len(ac))
        vacf[:n_valid] += ac[:n_valid]
        counts[:n_valid] += np.arange(len(ac), 0, -1)[:n_valid]

    vacf /= counts
    return vacf

# ───────────────────────────── PER-SET ────────────────────────────

def process_velocity_file(fname: str, fit_pars: Dict[int, Tuple[float, float]], ltt_time: np.ndarray):
    start_time = time.time()
    set_no = int(Path(fname).stem.split("-")[-1])
    if set_no not in fit_pars:
        print(f"  ⏭  {fname}: no fit parameters.")
        return None

    D_alpha, alpha = fit_pars[set_no]

    with h5py.File(fname, "r") as f:
        vx = f["trajectoryX"][:]
        vy = f["trajectoryY"][:]
        t_ps = f["time"][:]

    N_keep = int(len(t_ps) * KEEP_FRACTION)
    vx, vy, t_ps = vx[:N_keep], vy[:N_keep], t_ps[:N_keep]
    n_lag = min(MAX_LAG_FRAMES, N_keep)

    vacf = truncated_vacf(vx, vy, n_lag)
    ltt  = D_alpha * alpha * (alpha - 1) * ltt_time ** (alpha - 2)
    tau  = float((D_alpha / vacf[0]) ** (1 / (2 - alpha)))

    print(f"  ✔ {Path(fname).name:<30} {time.time() - start_time:5.2f} s")
    return t_ps[:n_lag], vacf, ltt, tau, set_no

# ───────────────────────────── TEMPERATURE LOOP ───────────────────

def run_prefix(prefix: str):
    fit_file = f"{prefix}-fit_parameters_10_80ps.txt"
    if not Path(fit_file).exists():
        print(f"❌ Missing {fit_file} – skipping {prefix}")
        return

    vel_files = sorted(Path().glob(f"{prefix}-lipid-velocity-*.h5"), key=lambda p: int(p.stem.split("-")[-1]))
    if not vel_files:
        print(f"❌ No velocity files for {prefix}")
        return

    print(f"\n📂  T = {prefix} K  →  {len(vel_files)} files   (fit: {fit_file})")
    fit_pars = read_fit_parameters(fit_file)
    ltt_time = np.arange(*LTT_RANGE_PS, dtype="float32")

    results = [r for r in (process_velocity_file(str(vf), fit_pars, ltt_time) for vf in vel_files) if r]
    if not results:
        return

    min_len = min(len(r[0]) for r in results)
    time_arr = results[0][0][:min_len]
    vacf_sets = np.stack([r[1][:min_len] for r in results])
    ltt_sets  = np.stack([r[2] for r in results])  # same length for all
    tau_vals  = np.array([r[3] for r in results])
    set_nos   = [r[4] for r in results]

    # ---------- write files ----------
    for vacf_i, s_no in zip(vacf_sets, set_nos):
        np.savetxt(f"vacf_T{prefix}_set{s_no}.txt", np.column_stack((time_arr, vacf_i)), header="Time(ps)  VACF", fmt="%.12f")

    for ltt_i, s_no in zip(ltt_sets, set_nos):
        np.savetxt(f"LTT_T{prefix}_set{s_no}.txt", np.column_stack((ltt_time, ltt_i)), header="Time(ps)  LTT", fmt="%.12f")

    np.savetxt(f"avg_vacf_xy_T{prefix}.txt", np.column_stack((time_arr, vacf_sets.mean(0), vacf_sets.std(0))), header="Time(ps)  Avg_VACF  Std_VACF", fmt="%.12f")
    np.savetxt(f"avg_LTT_fitting_T{prefix}.txt", np.column_stack((ltt_time, ltt_sets.mean(0))), header="Time(ps)  Avg_LTT", fmt="%.12f")

    # ---------- console summary ----------
    print(f"\nT = {prefix} K — τ values per set:")
    for s_no, tau in zip(set_nos, tau_vals):
        print(f"  Set {s_no}: τ = {tau:.4f} ps")
    print(f"✅ T={prefix} K → Mean τ = {tau_vals.mean():.4f} ps, Std = {tau_vals.std():.4f} ps\n")

# ───────────────────────────── MAIN ───────────────────────────────
if __name__ == "__main__":
    temps = sorted({p.stem.split("-")[0] for p in Path().glob("*-lipid-velocity-*.h5")})
    if not temps:
        print("No velocity files found in the current directory.")
    else:
        print("Detected temperatures:", ", ".join(temps))
        for T in temps:
            run_prefix(T)

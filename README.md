# GLE_framework

**Generalized Langevin Equation Study of Lipid Subdiffusion in Biomembrane Phases**

A collection of Python tools for analyzing lipid dynamics—particularly subdiffusion—via MSD, VACF, and memory-function approaches based on GLE.

---

## 📘 Overview

This project focuses on structural and dynamical analysis of lipid trajectories from MD simulations, providing:

- **Mean squared displacement (MSD)** → anomalous diffusion exponent (α) & diffusion coefficient (Dα)  
- **Velocity autocorrelation function (VACF)** using α and Dα as input  
- **Memory function** derived via the Generalized Langevin Equation (GLE)

---

## 🗂️ Project Structure

```text
GLE_framework/
├── 1_convert_xvg_h5.py   # Convert .xvg trajectory files to HDF5
├── 2_MSD.py              # Compute MSD, fit α & Dα
├── VACF.py               # Compute VACF using velocity data + α, Dα
├── MF.py                 # Derive memory/friction kernel via GLE
├── LICENSE               # MIT License
└── README.md             # This documentation

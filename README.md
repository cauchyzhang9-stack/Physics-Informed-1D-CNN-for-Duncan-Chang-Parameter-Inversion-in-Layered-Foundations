# Physics-Informed 1D-CNN for Duncan-Chang Parameter Inversion in Layered Foundations

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview
This repository contains the official source code for the paper **"Physics-Informed 1D-CNN for Duncan-Chang Parameter Inversion in Layered Foundations"** (Currently under peer review at *Computers & Geosciences*). 

The repository provides a complete workflow, including automated finite element (FE) synthetic data generation, the Fortran User Material (UMAT) subroutine for the Duncan-Chang constitutive model, and the implementation of a Physics-Informed 1D-CNN framework with a dynamic annealing training strategy.

## Repository Structure & File Description
The repository consists of individual source files organized as follows:

* **`abq_dc_v9.py`**: The automated Abaqus Python API script. It dynamically updates geometric/material parameters, dispatches FE simulations for dual-layer foundation Plate Load Tests (PLT), and extracts the macroscopic load-settlement (p-s) curves.
* **`duncan_solver.f`**: The Fortran User Material (UMAT) subroutine used in Abaqus to define the non-linear, stress-dependent Duncan-Chang hyperbolic constitutive model.
* **`cnnpinn2g4_annealing.py`**: **[Main Model]** The PyTorch implementation of the proposed Physics-Informed 1D-CNN. It features a dual-branch architecture and an archetype-dependent physical penalty loss dynamically guided by an annealing algorithm.
* **`cnn_late_clip_g3.py`**: Ablation study script. Implements a purely data-driven (blind) 1D-CNN that relies solely on post-processing hard physical boundary truncation ("late clipping").
* **`xgboost_late_clip_g3.py`**: Ablation study script. Implements a gradient boosting tree baseline (XGBoost) combined with post-processing late clipping.

## Dependencies and Requirements
To run the deep learning models and data processing scripts, the following Python environment is required:
* Python 3.8 or higher
* `torch` (PyTorch)
* `xgboost`
* `pandas`, `numpy`, `scipy`
* `scikit-learn`
* `matplotlib`, `tqdm`

To execute the data generation script (`abq_dc_v9.py`), you need:
* SIMULIA Abaqus (Tested on Abaqus 2020+)
* Intel Fortran Compiler (properly linked with Abaqus to compile the `.f` UMAT subroutine)

## Quick Test & Usage Instructions

### 1. Data Generation (Optional)
If you wish to generate new synthetic data from scratch:
1. Ensure your Abaqus command-line environment is properly configured with Fortran.
2. Place `abq_dc_v9.py` and `duncan_solver.f` in your Abaqus working directory.
3. Run the automated generation via the Abaqus command line:
   ```bash
   abaqus cae noGUI=abq_dc_v9.py
Source code for the physics-informed 1D-CNN framework used for Duncan-Chang parameter inversion in dual-layer foundations.

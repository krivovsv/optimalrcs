# optimalrcs: Nonparametric Reaction Coordinate Optimization with Histories

*A Python library for identifying optimal reaction coordinates in rare event dynamics.*

`optimalrcs` is a Python library for computing optimal reaction coordinates (RCs) from trajectory data using a nonparametric framework. It is specifically designed for studying rare event dynamics, where datasets are imbalanced and the configuration space is not extensively sampled.

The library supports both equilibrium and non-equilibrium dynamics and is robust to irregular or incomplete datasets. It incorporates trajectory histories to compensate for missing variables and uses stringent validation criteria, $Z_q$ and $Z_\tau$—to ensure that the computed RCs are optimal across time scales.

It was developed for arbitrary Markovian dynamics, including systems governed by diffusion processes with spatially varying diffusion coefficients and underdamped Langevin equations. See the accompanying manuscript for the exact mathematical formulation and theoretical background.


Key features:
- Supports both equilibrium and non-equilibrium dynamics
- Handles irregular and incomplete datasets
- Incorporates trajectory histories to compensate for missing variables
- Validated using stringent criteria: $Z_q$ and $Z_\tau$
- Designed for rare event dynamics with imbalanced datasets and limited sampling
- Applicable to arbitrary Markovian systems, including those with non-uniform diffusion and underdamped dynamics


## Installation

### Install directly from github repository

```bash
pip install git+https://github.com/krivovsv/optimalrcs.git
```
### Or clone and install locally

```bash
git clone https://github.com/krivovsv/optimalrcs.git
cd optimalrcs
pip install -e .
```

## Usage

Example: Optimizing the MFPT reaction coordinate for a protein folding trajectory using RMSD
from the native structure as a single input variable. Trajectory histories are used to
improve robustness and compensate for missing information.

```python
import optimalrcs

# Define a function that returns the collective variable (CV) time-series
def comp_y(): return rmsd

# Define history time delays to incorporate past trajectory information
history = [0,] + [2**i for i in range(9)]

# Initialize the MFPT optimizer with a boundary condition
mfpt=optimalrcs.MFPTNE(boundary0 = rmsd < 1.0)

# Run the optimization using the CV and history
mfpt.fit_transform(comp_y, history_delta_t = history, gamma = 0.1, max_iter=100000)

# Visualize the free energy profile and validation metrics
mfpt.plots_feps()
mfpt.plots_obs_pred()
```

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Reference

The methodology implemented in this library is described in the following paper:

Banushkina, P. V., & Krivov, S. V. (2025). *Nonparametric Reaction Coordinate Optimization with Histories: A Framework for
Rare Event Dynamics*. arXiv:XXXX.XXXXX

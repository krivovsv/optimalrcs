# optimalrcs: Nonparametric Reaction Coordinate Optimization with Histories

`optimalrcs` is a Python library for computing optimal reaction coordinates (RCs) from trajectory data using a nonparametric framework. It is specifically designed for studying rare event dynamics, where datasets are imbalanced and the configuration space is not extensively sampled.

The library supports both equilibrium and non-equilibrium dynamics and is robust to irregular or incomplete datasets. It incorporates trajectory histories to compensate for missing variables and uses stringent validation criteria, Z_q and Z_t, to ensure that the computed RCs are optimal across time scales.

## Installation

### install directly from github repository

```bash
pip install git+https://github.com/krivovsv/optimalrcs.git
```
### or clone and install locally

```bash
git clone https://github.com/krivovsv/optimalrcs.git
cd optimalrcs
pip install -e .
```

## Usage

Optimization of the mfpt RC using histories for the protein folding trajectory using rmsd form the native structure as a single input variable.

```python
def comp_y(): return rmsd
history = [0,] + [2**i for i in range(9)]
mfpt=optimalrcs.MFPTNE(boundary0 = rmsd < 1.0)
mfpt.fit_transform(comp_y, history_delta_t = history, gamma = 0.1, max_iter=100000)
mfpt.plots_feps()
mfpt.plots_obs_pred()
```

## License

This project is licensed under the MIT License. See the LICENSE file for details.
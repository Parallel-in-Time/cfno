# Chebyshev Fourier Neural Operators (CFNO)

_:scroll: Extension of the [Fourier Neural Operators](https://arxiv.org/abs/2010.08895) to PDE problems solved using pseudo-spectral (Chebyshev) space discretization methods._

## Content

- [cfno](./cfno/) : base Python module for CFNO
- [dedalus](./dedalus/) : scripts to run RBC simulations with Dedalus and pySDC
- [docs](./docs/) : some documentations about the FNO
- [script](./scripts/) : scripts for the full training pipeline (data generation, training, evaluation)
- [utils](./utils/) : utility scripts for cluster run

## Installation

In this folder, run this command to install `cfno` in your environment :

```bash
pip install -e .
```

> The `-e` option installs in _editable_ mode, which means any modification in the code won't need a re-installation to take the change into account.

> You can also use the `--user` option with `pip` to install without admin rights.

**Additional dependencies :**

- `dedalus` : spectral discretization for RBC. Recommended installation approach: [build from source.](https://dedalus-project.readthedocs.io/en/latest/pages/installation.html#building-from-source)

- `pySDC` : base package for SDC, need to be installed using a development version available in the `neuralpint` branch of its [main Github repo](https://github.com/Parallel-in-Time/pySDC/tree/neuralpint). To do that :

```bash
# Somewhere in a root folder ...
git clone https://github.com/Parallel-in-Time/pySDC.git
cd pySDC
git switch cfno
pip install -e .
```

Some changes may happen regularly on the development branch, to update your own version simply do

```bash
# In the pySDC repo
git pull
```

## How to use the code

See the full pipeline description in [scripts](./scripts/README.md). In particular, the main code parts it uses are :

- [cfno.models.cfno2d](./cfno/models/cfno2d.py) : implementation of the CFNO model
- [cfno.losses](./cfno/losses) : module for the different losses
- [cfno.training.pySDC](./cfno/training/pySDC.py) : base `FourierNeuralOp` class used for training and / or inference
- [cfno.data.preprocessing](./cfno/data/preprocessing.py) : `HDF5Dataset` class and `createDataset` function used to create training datasets

## Acknowledgements

This project has received funding from the [European High-Performance
Computing Joint Undertaking](https://eurohpc-ju.europa.eu/) (JU) 
under grant agreement No 101118139 ([Inno4Scale - NeuralPint](https://www.inno4scale.eu/neuralpint/)).
The JU receives support from the European Union's Horizon 2020 research
and innovation programme and Germany.

<p align="center">
  <img src="./docs/img/EuroHPC.jpg" height="105"/> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <img src="./docs/img/LogoInno4Scale.png" height="105" /> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <img src="./docs/img/BMBF_gefoerdert_2017_en.jpg" height="105" />
</p>
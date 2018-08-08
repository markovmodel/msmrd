# MSM/RD
## Coupling Markov state models of molecular kinetics with reaction-diffusion simulations.

This project was born with the desire to couple Markov state models (MSM) derived from Molecular dynamics simulations with particle-based reaction diffusion (RD) simulations. The main goal is to obtain the best of both worlds by efficiently producing simulations at large time- and lengthscales, while still conserving molecular resolution.

The package in this repository is proof of concept of MSM/RD. It implements the methodology presented in [the MSM/RD paper](https://doi.org/10.1063/1.5020294), and it was used to produce most of the figures and results therein. A more general framework and implementation is currently under development.

## Installation
```
git clone https://github.com/markovmodel/msmrd.git
cd msmrd
python setup.py install
```

## Software dependencies
- [PyEMMA 2.4](http://emma-project.org/latest/)
- Python 2.7 
- Numpy 1.12
- Cython 0.25
- Jupyter 4.1
- Matplotlib 2.0
- Scipy 0.19
- Dill 0.2 
- Multiprocess 0.7

All these packages are available through the [conda environment managment](https://conda.io/docs/).

## Notebooks
A set of Jupyter notebooks can be found in `scripts/current/`. These were used to produce the results in the paper. The data files are too large and thus not available in the github repository. However, if you are interested in this, plese feel free to contact us.

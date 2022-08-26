# Asynchronous Bayesian Optimisation with Selective Killing

Please note a large proportion of this repository is imported from AEGiS found at https://github.com/LewisADay/aegisk. Notably this readme is largly simply an adapted form of the reasme from that repository to make the appropriate changes for this work.

This repository contains the Python3 code for the experiments and results presented as my maters project in report and presentation form. All setup instructions are inherited from the submodule AEGiS as presented in:

> George De Ath, Richard M. Everson, and Jonathan E. Fieldsend. Asynchronous ϵ-Greedy Bayesian Optimisation. Proceedings of the Thirty-Seventh Conference on Uncertainty in Artificial Intelligence, PMLR 161:578-588, 2021. </br>
> **Paper**: https://proceedings.mlr.press/v161/de-ath21a.html

The repository also contains all training data used for the initialisation of
the optimisation runs carried out, the optimisation results of each of the
runs, and jupyter notebooks to generate the results, figures and tables in the
paper.

## Installation Instructions

```script
> git clone https://github.com/LewisADay/aegisk
> cd aegisk
> conda create -n aegisk python=3.7 numpy matplotlib scipy statsmodels tqdm docopt
> conda activate aegisk
> conda install pytorch cpuonly botorch pytorch -c pytorch -c gpytorch
> pip install pyDOE2 pygmo==2.13.0 pygame box2d-py
```

- Profet benchmark (optional):
```script
> pip install pybnn
> pip install git+https://github.com/amzn/emukit
```

## Optimisation results

The results of all optimisation runs can be found in the `results` directory.
The notebook [comparisons.ipynb](comparisons.ipynb) shows how to a load all
the experimental data and to subsequently plot it.

## Reproduction of figures and tables in the paper

- [comparisons.ipynb](comparisons.ipynb) contains the code to load
and process the optimisation results (stored in `results` directory), as well
as the code to produce all results figures and tables used in the report.
- [presentation_plots.ipynb](presentation_plots.ipynb) contains the code to
create the figures used in the presentation.

## Training data

The initial training locations for each of the 21 sets of
[Latin hypercube](https://www.jstor.org/stable/1268522) samples for the various noise levels are located in the `data` directory. The files are named like `ProblemName_number.pt`, e.g. first set of training locations for the Branin problem is stored in `Branin_001.pt`. Each of these files is a compressed numpy file created with [torch.save](https://pytorch.org/docs/stable/torch.html#torch.save). It has two [torch.tensor](https://pytorch.org/docs/stable/torch.html#torch.tensor) arrays (`Xtr` and `Ytr`) containing the 2*D initial locations and their corresponding fitness values. Note that for problems that have a non-default dimensionality (e.g. Ackley with d=5), then the data files have the dimensionality appended, e.g. `Ackley5_001.pt`; see the suite of [available synthetic test problems](aegis/test_problems/synthetic_problems.py). To load and inspect the training data, use the following instructions:

```python
> python
>>> import torch
>>> data = torch.load('data/Ackley5_001.pt')
>>> Xtr = data['Xtr']  # Training data locations
>>> Ytr = data['Ytr']  # Corresponding function values
>>> Xtr.shape, Ytr.shape
(torch.Size([10, 5]), torch.Size([10]))
```

# Neural Operator with Regularty Structure

This repository contains the code for the Neural Operator with Regularity Structure (NORS). 

Stochastic partial differential equations (SPDEs) are significant tools for modelling dynamics in many areas.  As Neural Operators, strong tools for solving parametric PDEs, lack the ability to modeling stochastic PDEs which usually have poor regularity due to the driving noise, we propose the Neural Operator with Regularity Structure (NORS) which incorporates the feature vectors for modeling dynamics driven by SPDEs. 

We conduct experiments on the dynamic Φ41 model, the parabolic equation with multiplicative forcing and the 2d stochastic Navier-Stokes equation, and the results demonstrate that the NORS is resolution-invariant, efficient, and can achieve one order of magnitude lower error with a modest amount of data.

## Files

- `Classes`: 
  - `Noise.py` (generates noises and initial conditions for the numerical experimants), 
  - `SPDEs.py` (SPDE solver of the Φ41 equations and the parabolic equation with multiplicative forcing using finite difference method, integrator functions for the Model class), 
  - `Rule.py` (helper class that generates the rule for creating the model feature set)，
  - `Model.py` (generates model features).
- `Data`: conventional solvers used to generate the datasets for the 2d stochastic Navier-Stokes equation.
- `NORS_1d_time.py`: the NORS for 1d spatial + 1d temporal equation problem such as the Φ41 equation and the parabolic equation with multiplicative forcing.
- `NORS_2d.py`: the NORS for 2d spatial equation problem such as the 2d stochastic Navier-Stokes equation.
- `model_xxx.py`: generates the model feature vectors.
- `phi41_time.py`, `multi_time.py`, `NS.py`: trains and evaluates an NORS on three SPDEs.

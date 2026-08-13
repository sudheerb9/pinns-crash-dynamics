# PINNs for Crash Dynamics

Physics-informed neural networks for **head-on vehicle crash** modeling — predict displacement, velocity, and acceleration during impact without re-running a full FEM job for every design tweak.

Frontal barrier impact is represented with lumped-parameter **1-DOF** and **2-DOF** spring–mass–damper models. The network is trained so ODE residuals (and initial conditions) enter the loss; after training, kinematics evaluate essentially instantly. Against analytical solutions and ANSYS Explicit Dynamics baselines, error stays in a range useful for early screening (on the order of ≤ 18% in the reported comparisons). **Transfer learning** reuses a trained source model as the starting point for new structural parameter sets, so variants need far less training than starting from scratch.

Stack: **Python**, **TensorFlow** (primary), with PyTorch notebooks and an optional Julia/NeuralPDE prototype.

> Developed as B.Tech work at IIT Bhubaneswar (2022). Write-up and code appendix: [`docs/`](docs/).

## What’s included

```
src/
  pinn_1dof.py             # 1-DOF TensorFlow PINN
  pinn_2dof.py             # 2-DOF TensorFlow PINN
  transfer_learning.py     # Source → target weight reuse
notebooks/
  pinn_solver_tensorflow.ipynb   # TF solver + inverse / parameter ID
  btp_v1_1dof_pytorch.ipynb      # PyTorch 1-DOF experiments
  btp_v2_2dof_pytorch.ipynb      # PyTorch 2-DOF experiments
julia/
  btp_neuralpde.jl         # NeuralPDE prototype
docs/                      # Thesis PDF (+ version with code appendix)
figures/                   # PINN vs analytical / FEM plots
```

## Approach

1. Embed the crash ODEs in the PINN loss; sample ~10⁴ collocation points in time.
2. Train with Adam and Glorot initialization; read displacement from the network, then get velocity / acceleration by differentiation.
3. For a new vehicle variant, load source weights and fine-tune instead of training from random init.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cd src
python pinn_1dof.py          # 1-DOF
python pinn_2dof.py          # 2-DOF
python transfer_learning.py  # source → structural variant
```

Interactive workflows (including inverse PINN / parameter identification) live under `notebooks/`.

## Results

See `figures/` for the comparison plots:

- PINN vs analytical 1-DOF / 2-DOF displacement and velocity
- PINN vs ANSYS Explicit Dynamics deformation and equivalent stress
- Responses after transfer learning on changed structural parameters

The car–wall FEM explicit runs took ~**25 hours**; the trained PINN returns kinematics in near real time for screening.

# PINNs for Crash Dynamics

B.Tech thesis project (IIT Bhubaneswar, 2022) — **Physics-Informed Neural Networks** for modeling head-on vehicle crash dynamics.

Predicts **stress / velocity / displacement** during impact from lumped-parameter 1-DOF and 2-DOF spring–mass–damper models, with near-instant inference after training versus ~25-hour FEM (ANSYS Explicit Dynamics) runs. Validated against analytical solutions and FEM baselines (thesis: typically within ~18% for screening use). **Transfer learning** reuses trained weights across structural variants to cut retraining cost.

| | |
|---|---|
| Author | Bulusu Sri Datta Sudheer (`18ME01017`) |
| Advisor | Dr. B Pattabhi Ramaiah |
| Stack | Python, TensorFlow (primary), PyTorch (exploratory), Julia/NeuralPDE (optional) |

## Repository layout

```
src/
  pinn_1dof.py            # Thesis appendix: 1-DOF TensorFlow PINN
  pinn_2dof.py            # Thesis appendix: 2-DOF TensorFlow PINN
  transfer_learning.py    # Weight reuse across structural variants
notebooks/
  pinn_solver_tensorflow.ipynb   # TF solver + inverse/parameter ID classes
  btp_v1_1dof_pytorch.ipynb      # PyTorch 1-DOF experiments
  btp_v2_2dof_pytorch.ipynb      # PyTorch 2-DOF experiments
julia/
  btp_neuralpde.jl        # NeuralPDE / Flux prototype
docs/
  18ME01017_BTP_thesis.pdf
  18ME01017_BTP_final_with_code_appendix.pdf
figures/                  # PINN vs analytical / FEM comparison plots
```

## Method (short)

Frontal barrier crash is modeled as free vibration of damped spring–mass systems (literature lumped-parameter models). The PINN embeds the ODE residuals in the loss, collocates ~10⁴ time points, and trains with Adam + Glorot init. Displacement comes from the network; velocity and acceleration follow by differentiation.

Transfer learning: train a **source** vehicle parameter set, then initialize a **target** variant from those weights and fine-tune (thesis Fig. 4.3).

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 1-DOF / 2-DOF training (TensorFlow)
cd src
python pinn_1dof.py
python pinn_2dof.py

# Transfer learning demo (source → structural variant)
python transfer_learning.py
```

Open the notebooks under `notebooks/` for the original interactive workflows (including inverse PINN / parameter identification in `pinn_solver_tensorflow.ipynb`).

## Results snapshot

Thesis comparisons (see `figures/` and `docs/`):

- PINN displacement / velocity vs analytical 1-DOF and 2-DOF solutions
- PINN vs ANSYS Explicit Dynamics deformation and equivalent stress
- Transfer-learning responses after changing structural parameters

FEM explicit crash jobs on the car–wall model took on the order of **25 hours**; the trained PINN evaluates kinematics essentially instantly for early design screening.

## Citation

If you use this work, please cite the thesis:

> Bulusu Sri Datta Sudheer. *Transfer Learning Techniques to Solve Transient Non-Linear Partial Differential Equations*. B.Tech thesis, School of Mechanical Sciences, IIT Bhubaneswar, May 2022.

Key references used in the project are listed in `docs/18ME01017_BTP_thesis.pdf` (Raissi et al. PINNs; Munyazikwiye et al. vehicle crash lumped models).

## License

Code and figures from this thesis project are shared for portfolio / educational use. The thesis PDFs remain subject to IIT Bhubaneswar academic use norms.

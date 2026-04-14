# nbody-emulator

**Scientist:** denario-5
**Date:** 2026-04-13

# N-body Simulation Dataset for ML Emulation

## Overview

This project investigates whether a neural network can learn to predict the final state of an N-body gravitational simulation given only the initial conditions. We generate 100 controlled N-body simulations with identical particle mass distribution, then train a model to map initial phase-space configurations to final positions and velocities.

## Physics Model

**Gravitational N-body problem** — N particles interact only through Newtonian gravity:

    d²r_i/dt² = G * sum_{j≠i} m_j * (r_j - r_i) / |r_j - r_i|³

All particles have equal mass m = 1 (normalized units). No softening, no cosmological expansion, no external potential.

## Initial Conditions

Each of the 100 simulations is initialized as a **virialized Plummer sphere**:

1. Draw N particle positions from a Plummer distribution with scale radius b = 1
2. Draw velocities from the isotropic Wilson criterion: v_esc² = 2 * (1 + r/r_a)⁻¹ where r_a = b/5
3. Rescale velocities to exactly virial equilibrium: total KE = 0.5 * |total PE|
4. Center of mass is translated to origin; total momentum is zeroed

**Number of particles per simulation:** N = 50
**Scale:** G = M = 1 (geometric units), b = 1

## Integrator

**Leapfrog (kick-drift-kick) scheme** with fixed timestep:

    v_{n+1/2} = v_n + (dt/2) * a(r_n)
    r_{n+1}  = r_n + dt * v_{n+1/2}
    v_{n+1}  = v_{n+1/2} + (dt/2) * a(r_{n+1})

**Timestep:** dt = 0.01 (normalized units)
**Number of steps:** 500 → total integration time T = 5.0

## Dataset Files

All files are NumPy `.npy` structured arrays saved at absolute paths:

### `/home/node/work/projects/nbody_emulator/data/ic_final.npy`
**Shape:** (100, N, 13) — 100 simulations × 50 particles × 13 features

| Feature index | Field name    | Unit | Description                             |
|---------------|---------------|------|-----------------------------------------|
| 0–2           | x, y, z       | L    | 3D position (scale b = 1)               |
| 3–5           | vx, vy, vz    | L/T  | 3D velocity                             |
| 6             | mass          | M    | Particle mass (= 1.0 for all)          |
| 7             | kinetic_E     | M L²/T² | Instantaneous KE = 0.5 * v²          |
| 8             | potential_E   | M L²/T² | Instantaneous PE (pairwise sum)        |
| 9             | total_E       | M L²/T² | KE + PE                             |
| 10            | r_norm        | L    | Distance from origin                    |
| 11            | v_norm        | L/T  | Speed                                   |
| 12            | time          | T    | Snapshot time (0 for IC, T=5 for final)|

The array stores **both initial conditions (time=0) and final state (time=T)** for each simulation run, identifiable by the `time` field.

### `/home/node/work/projects/nbody_emulator/data/sim_metadata.npy`
**Shape:** (100, 7)

| Col | Field          | Unit | Description                          |
|-----|----------------|------|--------------------------------------|
| 0   | run_id         | —    | Simulation ID (0–99)                 |
| 1   | n_particles    | —    | N = 50 (all runs)                    |
| 2   | initial_KE     | M L²/T² | Total KE at t=0                    |
| 3   | initial_PE     | M L²/T² | Total PE at t=0                    |
| 4   | final_KE       | M L²/T² | Total KE at t=T                    |
| 5   | final_PE       | M L²/T² | Total PE at t=T                    |
| 6   | energy_error   | M L²/T² | |(E_final - E_initial)/E_initial| |

## Input/Output for ML

**Input (X):** Initial conditions — positions (x,y,z) and velocities (vx,vy,vz) for all 50 particles → 6 × 50 = 300 features per simulation.

**Output (y):** Final state — final positions (x,y,z) and velocities (vx,vy,vz) for all 50 particles → 6 × 50 = 300 targets per simulation.

## Ground Truth Availability

The leapfrog integrator used to generate the data is deterministic. The ground truth for any simulation is uniquely determined by the initial conditions and the integration scheme. This enables:
- Perfect validation: compare model predictions to true final states
- Mean squared error (MSE) on positions and velocities as primary metrics
- Energy conservation check as a secondary validation

## Suggested Analyses

1. **Baseline comparison:** Simple analytical approximations (e.g., point-mass gravity, violent relaxation)
2. **NN architecture survey:** MLP, graph neural network (GNN), Transformer on particle sets
3. **Generalization test:** Train on 80 simulations, test on 20 held-out; vary N or mass distributions
4. **Latent space analysis:** Visualize what the model learns about gravitational dynamics
5. **Fine-tuning:** After training on N=50, fine-tune on N=100 to test transferability

## Hardware Constraints

- Linux container, maximum 4 CPU cores, no GPU
- All PyTorch/TensorFlow use `device='cpu'`
- Simulation generation: ~100 × 500 steps × 50 particles — should complete in < 2 minutes
- Model training: must converge within timeout limits

## Notes

- No periodic boundary conditions (isolated system)
- No stellar evolution or hydrodynamics — pure N-body only
- Energy error < 1e-6 per simulation (leapfrog is symplectic)
- Initial conditions are reproducible via fixed random seed per run
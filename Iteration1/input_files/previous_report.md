

Iteration 0:
### Summary: Symplectic Neural ODE for N-Body Emulation

**Project Status:** Completed initial proof-of-concept.
**Core Objective:** Emulate $t=5.0$ final states of $N=50$ virialized Plummer spheres using a Symplectic Neural ODE.

**Key Findings:**
*   **Architecture:** A permutation-invariant Interaction Network (GNN) parameterizing the vector field $\dot{z} = f(z, \theta)$ outperformed a baseline MLP by an order of magnitude in MSE.
*   **Integration:** Replacing standard RK4 with a differentiable leapfrog (Verlet) integrator was critical for maintaining Hamiltonian structure and bounding energy drift to <1%.
*   **Physical Priors:** Explicitly encoding Newton’s Third Law ($f_{ij} = -f_{ji}$) and the gravitational softening kernel ($\epsilon=0.01$) significantly stabilized training and improved accuracy.
*   **Training Strategy:** Training on 10 intermediate snapshots ($\Delta t = 0.5$) provided necessary gradient supervision, preventing the "smearing" effect observed in direct $t=0 \to t=5$ regression.
*   **Ablation:** Removing the softening kernel ($\epsilon=0$) caused numerical instability and exploding gradients due to $1/r^2$ singularities, confirming that physical priors are essential for convergence.

**Constraints & Decisions:**
*   **Data:** 100 simulations (80 train/20 test). Input features normalized by simulation-specific virial radius and velocity dispersion.
*   **Augmentation:** Rotational and Galilean invariance enforced via on-the-fly 3D rotations/reflections.
*   **Regularization:** Dynamic weighting ($\lambda$) of the energy conservation penalty allowed the model to prioritize trajectory reconstruction before enforcing Hamiltonian constraints.
*   **Hardware:** Training performed on GPU (NVIDIA RTX PRO 6000) using mixed precision.

**Limitations & Uncertainties:**
*   **Scalability:** Model performance on $N > 50$ remains untested (though GNN architecture is theoretically size-agnostic).
*   **Complexity:** Current model assumes equal particle masses ($m=1$).
*   **Scope:** Isolated system only; no periodic boundary conditions or cosmological expansion.

**Future Directions:**
*   **Transferability:** Test zero-shot scaling to $N=100$ or $N=1000$.
*   **Generalization:** Introduce mass variance to evaluate learning of mass-dependent dynamics.
*   **Cosmology:** Adapt integrator for comoving coordinates and periodic boundaries to support large-scale structure emulation.
        
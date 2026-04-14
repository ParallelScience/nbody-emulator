1. **Data Preprocessing and Augmentation**:
   - Load the `ic_final.npy` dataset. Re-run the leapfrog integrator to generate 10 intermediate snapshots at $t \in \{0.5, 1.0, \dots, 4.5\}$ and save these to a new file to avoid redundant computation.
   - Apply data augmentation: perform random 3D rotations and reflections on the particle coordinates and velocities for each simulation to exploit the rotational and Galilean invariance of the gravitational system.
   - Normalize inputs by the virial radius and velocity dispersion of each individual simulation to ensure scale-invariant learning.

2. **Interaction Network Architecture**:
   - Implement a permutation-invariant GNN where each particle is a node.
   - Explicitly enforce Newton’s Third Law ($f_{ij} = -f_{ji}$) in the edge function to ensure action-reaction symmetry.
   - Incorporate the gravitational softening $\epsilon = 0.01$ as a fixed physical prior in the pairwise interaction calculation.
   - Use sparse adjacency operations to compute interactions efficiently, minimizing memory overhead for the CPU-only environment.

3. **Symplectic Neural ODE Formulation**:
   - Define the system dynamics $\dot{z} = f(z, \theta)$ parameterized by the GNN.
   - Replace the standard RK4 solver with a **Verlet-based (leapfrog) integrator** within the Neural ODE framework to preserve the Hamiltonian structure of the N-body system and reduce numerical dissipation.

4. **Physics-Informed Loss Function**:
   - Define the primary loss as the MSE between predicted and ground truth positions/velocities at all intermediate snapshots.
   - Implement a dynamic weighting strategy for the energy regularization term: start with a low $\lambda$ to prioritize trajectory reconstruction, then gradually increase $\lambda$ to enforce Hamiltonian conservation.
   - Calculate the energy penalty $|E_{pred} - E_{initial}|$ at every intermediate step of the ODE solver to ensure the model remains on the energy manifold throughout the integration.

5. **Training Protocol**:
   - Split the 100 simulations into a training set (80 runs) and a validation set (20 runs).
   - Use the Adam optimizer with a learning rate scheduler.
   - Add a small amount of Gaussian noise to the input initial conditions during training to improve the robustness of the learned vector field and prevent overfitting.

6. **Computational Efficiency and Stability**:
   - Ensure efficient batching by processing all 100 simulations in parallel.
   - Monitor validation loss and energy conservation metrics to detect overfitting early.
   - Avoid unnecessary memory copies during the ODE integration steps to stay within CPU-only hardware constraints.

7. **Model Evaluation and Benchmarking**:
   - Evaluate the model's ability to reconstruct the full trajectory by comparing the predicted $t=5$ state against the ground truth.
   - Compare the performance of the Symplectic Neural ODE against a baseline MLP that maps $t=0$ directly to $t=5$.

8. **Ablation and Final Analysis**:
   - Perform an ablation study by training a version of the model without the explicit softening kernel $\epsilon$ to quantify the impact of physical priors.
   - Calculate final MSE metrics on the held-out test set and visualize predicted vs. true trajectories to qualitatively assess the model's capture of gravitational dynamics.
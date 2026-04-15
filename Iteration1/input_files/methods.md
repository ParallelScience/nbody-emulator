1. **Data Preparation and Trajectory Generation**:
   - Load the `ic_final.npy` dataset. Re-run the leapfrog integrator to generate 10 intermediate snapshots at $t \in \{0.5, 1.0, \dots, 4.5\}$ for each of the 100 simulations.
   - Implement a data loader that supports dynamic particle masking to facilitate $N$-sensitivity analysis, allowing evaluation on subsets of the 50 particles.
   - Apply standard normalization to positions and velocities based on the initial virial radius and velocity dispersion of each simulation to ensure scale-invariance.

2. **Interaction Network Architecture**:
   - Implement a permutation-invariant GNN where each particle is a node.
   - Explicitly enforce Newton’s Third Law ($f_{ij} = -f_{ji}$) in the edge function.
   - Parameterize the softening kernel $\epsilon$ as a learnable scalar parameter initialized at 0.01 to test if the model can recover the physical prior.
   - Ensure the architecture is strictly CPU-compatible using standard PyTorch operations.

3. **Comparative ODE Frameworks**:
   - Implement two distinct Neural ODE variants: one using a symplectic Verlet-based (leapfrog) integrator and one using a standard non-symplectic RK4 solver.
   - Both models will share the exact same GNN backbone to isolate the impact of the integration scheme on energy conservation and long-term trajectory stability.

4. **Training Protocol and Long-Term Rollout**:
   - Train both models using only the first half of the trajectory ($t=0$ to $t=2.5$) as supervision.
   - Use the Adam optimizer with a fixed learning rate.
   - Perform "long-term rollout" evaluation at $t=5.0$ by using the learned vector field integrated by the chosen solver (Verlet or RK4) starting from the $t=0$ state, ensuring the model performs autonomous integration rather than using intermediate snapshots as warm starts.

5. **Energy Conservation Analysis**:
   - Train the models with the energy penalty $\lambda$ set to zero during the final evaluation phase to determine if the symplectic architecture inherently conserves the Hamiltonian.
   - Quantify energy drift by calculating the mean absolute energy error over the entire $t=0$ to $t=5$ interval and the rate of energy drift ($\Delta E / \Delta t$) for both the Symplectic and RK4 models to demonstrate the superiority of the symplectic manifold.

6. **Generalization and Sensitivity Analysis**:
   - Evaluate the trained models on held-out simulations (20 runs) and on subsets of particles (e.g., $N=25, 40$) to assess robustness to varying interaction topologies.
   - Quantify performance using MSE on positions and velocities at $t=5.0$ across different $N$ values.

7. **Ablation and Latent Parameter Analysis**:
   - Analyze the learned value of the softening parameter $\epsilon$ to determine if the model converges to the ground-truth value of 0.01.
   - Compare the performance of the model with the learned $\epsilon$ against a version where $\epsilon$ is fixed at 0.01 to quantify the benefit of learning the physical prior.

8. **Final Benchmarking**:
   - Compare the Symplectic Neural ODE, the RK4-based Neural ODE, and the baseline MLP (mapping $t=0$ to $t=5$) based on their ability to generalize to $t=5.0$ after being trained only on the $t=0$ to $t=2.5$ interval.
   - Visualize predicted vs. true trajectories for the best-performing model to qualitatively assess the capture of gravitational dynamics.
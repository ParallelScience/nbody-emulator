1. **Data Preprocessing and Snapshot Generation**:
   - Load the `ic_final.npy` dataset. Generate 10 intermediate snapshots at $t \in \{0.5, 1.0, \dots, 4.5\}$ for each of the 100 simulations using the provided leapfrog integrator.
   - Calculate the ground-truth acceleration $a_i$ and the velocity update $\Delta v_i$ over the interval $\Delta t = 0.01$ for each particle at every snapshot using the fixed softening parameter $\epsilon = 0.01$.
   - Normalize positions and velocities using global statistics derived from the initial conditions.

2. **GNN Architecture Design**:
   - Implement a permutation-invariant Interaction Network where each particle is a node.
   - Define the edge function to compute pairwise interactions: $f_{ij} = \phi(r_i, r_j, \epsilon)$, with $\epsilon = 0.01$ fixed.
   - Enforce Newton’s Third Law by ensuring the edge function outputs $f_{ij} = -f_{ji}$.
   - Aggregate edge outputs at each node to compute the total force/acceleration acting on particle $i$.

3. **Supervised Force-Field Training with Physics Constraints**:
   - Train the GNN to predict the velocity update $\Delta v_i$ (or acceleration $a_i$) at time $t$.
   - Define the loss function as the sum of: (a) MSE between predicted and ground-truth $\Delta v_i$, (b) a penalty term for the violation of linear momentum (sum of predicted forces $\approx 0$), and (c) a physics-informed regularization term penalizing local energy drift over a single step.

4. **Curriculum Learning Strategy**:
   - Begin training on $N=2$ and $N=3$ particle configurations generated with Plummer sphere physics.
   - Include a "warm-up" phase where the model is tested on intermediate $N$ (e.g., $N=10, 25$) before transitioning to the full $N=50$ dataset to ensure stable convergence.

5. **Inference via External Integration**:
   - During evaluation, use the trained GNN as a "learned force law" function.
   - Plug this function into a standard, non-differentiable leapfrog integrator to perform the $t=0$ to $t=5$ rollout.
   - This ensures numerical stability is handled by the symplectic solver while the GNN provides the learned dynamics.

6. **Energy Conservation and Physical Consistency**:
   - Monitor the total energy $E = KE + PE$ throughout the rollout.
   - Compare the energy drift of the GNN-integrated trajectory against the ground-truth leapfrog integration.
   - Use the `total_E` metadata to verify that the learned force field respects the Hamiltonian structure of the system.

7. **Generalization and Sensitivity Analysis**:
   - Evaluate the model on 20 held-out simulations to test generalization to unseen initial conditions.
   - Test performance on varying $N$ (e.g., $N=25, 40$) to confirm that the GNN's permutation invariance allows for flexible particle counts.

8. **Benchmarking and Performance Validation**:
   - Compare the GNN-based emulator against a baseline MLP (mapping $t=0$ to $t=5$).
   - Evaluate both models on their ability to generalize to different $N$ (e.g., $N=25, 40$) to demonstrate the GNN's architectural advantage.
   - Quantify improvements in physical consistency (energy conservation) and trajectory accuracy at $t=5.0$.
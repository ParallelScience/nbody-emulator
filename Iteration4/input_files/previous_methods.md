1. **Dataset Preparation and Lyapunov Characterization**:
   - Generate the 100 simulations using the leapfrog integrator.
   - Calculate the global Lyapunov time ($\lambda^{-1}$) by evolving two perturbed initial conditions (applying a global perturbation $\delta r \approx 10^{-8}$ to all particles). Use a high-precision integration method (or a significantly smaller timestep than 0.01) to ensure the measured divergence reflects the physical system's sensitivity rather than integrator truncation error.
   - Generate 10 intermediate snapshots ($t \in \{0.5, \dots, 4.5\}$) and compute the ground-truth acceleration $a_i$ for each particle at each snapshot.
   - Partition the data into training (80 runs), validation (10 runs), and testing (10 runs).

2. **GNN Architecture Design**:
   - Implement a permutation-invariant Interaction Network where the edge function $\phi$ takes relative positions ($r_j - r_i$) and relative velocities ($v_j - v_i$) as inputs.
   - Include a modular switch to toggle the inclusion of the analytical gravitational force $f_{\text{phys}}$ (with softening $\epsilon=0.01$) as an additional input feature to the edge MLP.
   - Enforce Newton’s Third Law by defining the edge output as $f_{ij} = \text{MLP}(r_{ij}, v_{ij})$ and ensuring the final force on particle $i$ is $\sum_{j \neq i} (f_{ij} - f_{ji}) / 2$.

3. **Training Protocol**:
   - Train the model to minimize the MSE between the predicted acceleration $\hat{a}_i$ and the ground-truth acceleration $a_i$ derived from the leapfrog integrator.
   - Utilize the training set to optimize model weights using the Adam optimizer.
   - Rely on the architectural constraints (Newton’s Third Law) and the MSE objective as the primary drivers for learning the dynamics.

4. **Ablation Study: Physics Prior vs. Learned Dynamics**:
   - Train two versions of the GNN: (A) "Physics-Informed" (receiving $f_{\text{phys}}$ as input) and (B) "Learned-Only" (receiving only raw kinematics).
   - Compare the performance of both models on the test set to determine if the model successfully captures gravitational dynamics without explicit analytical guidance.

5. **Hamiltonian Consistency Analysis**:
   - Test if the GNN learns a Hamiltonian flow by integrating the learned force field using both a symplectic integrator (leapfrog) and a non-symplectic integrator (RK4).
   - Use a small timestep (e.g., $dt = 0.001$) for the RK4 integration to ensure stability.
   - Define "Energy Drift" as the variance of the total energy over the integration time $T=5.0$. Compare the drift of the GNN-driven rollout against a baseline MLP-based force field to quantify the conservative nature of the learned vector field.

6. **N-Scaling Generalization Test**:
   - Generate new test sets with $N=25$ and $N=100$ particles using the same Plummer sphere initialization parameters.
   - Evaluate the trained $N=50$ model on these sets without retraining.
   - Quantify the degradation in MSE and energy conservation to assess scale-invariance and potential overfitting to the $N=50$ density distribution.

7. **Benchmarking and Performance Validation**:
   - Compare the GNN-based emulator against a baseline MLP that maps $t=0$ directly to $t=5$.
   - Evaluate the models based on their ability to track the trajectory within the calculated Lyapunov time.
   - Use the `total_E` metadata to verify that the GNN-driven rollout maintains physical consistency relative to the ground-truth leapfrog integration.

8. **Final Evaluation and Reporting**:
   - Aggregate results from the ablation study, Lyapunov analysis, and N-scaling tests.
   - Document the model's performance in terms of trajectory reconstruction error and energy conservation, explicitly distinguishing between physical divergence (tracking the Lyapunov time) and numerical model-induced error.
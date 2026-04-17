1. **Hamiltonian Neural Network (HNN) Architecture**:
    - Construct a GNN-based HNN that parameterizes the interaction potential $U(r) = \sum_{i<j} \phi(|r_i - r_j|)$, where $\phi$ is a shared MLP.
    - Define the kinetic energy as $T(v) = \sum \frac{1}{2} m_i v_i^2$ (with $m=1$).
    - Use automatic differentiation to derive forces: $\dot{p}_i = -\nabla_{r_i} U$.
    - Implement the GNN using vectorized matrix operations to compute pairwise distances and potential sums, ensuring efficiency on CPU hardware.

2. **Leapfrog-Integrated Training Loop**:
    - Implement a custom "kick-drift-kick" integration step within the training loop.
    - The "kick" step uses the force derived from the HNN potential gradient $\nabla_r U$ via `torch.autograd.grad`.
    - The "drift" step updates positions using the current velocities.
    - Ensure the loop is fully differentiable to allow backpropagation through the integration steps.

3. **Data Efficiency and Augmentation**:
    - Utilize a "sliding window" approach: train the model to predict $t_{n+1}$ from $t_n$ for all 10 intermediate snapshots per simulation, increasing the effective training sample size.
    - Apply random rotations and translations to the particle coordinates as a data augmentation strategy to exploit the system's inherent symmetries and increase configuration diversity.

4. **Curriculum Learning Strategy**:
    - Stage 1: Train the HNN on snapshots where particles are at larger radii ($r > 0.5b$) to learn the coarse-grained potential.
    - Stage 2: Gradually introduce snapshots containing the high-density core ($r < 0.5b$) to refine the model's ability to handle close-encounter dynamics.

5. **Physical Prior Integration (Softening)**:
    - Incorporate the gravitational softening $\epsilon=0.01$ directly into the pairwise distance calculation: $d_{ij} = \sqrt{|r_i - r_j|^2 + \epsilon^2}$.
    - Pass this softened distance to the MLP $\phi$ to allow the network to learn the softened potential surface directly, ensuring the model remains consistent with the ground-truth leapfrog integrator.

6. **Training and Optimization**:
    - Train the model by minimizing the MSE between the predicted trajectory snapshots and the ground-truth leapfrog snapshots.
    - Use the Adam optimizer with a learning rate scheduler to ensure convergence within CPU-only constraints.
    - Scale message-passing weights by $1/N$ to maintain stability across different particle counts.

7. **Validation and Symplectic Assessment**:
    - Evaluate the model using "Symplectic Drift": calculate the Hamiltonian deviation $\Delta H = |H(t) - H(0)|$ over the full $T=5.0$ integration.
    - Quantify time-reversibility error by integrating forward from $t=0$ to $t=5.0$ and backward to $t=0$, measuring the Euclidean distance to the original initial conditions.
    - Monitor phase-space volume preservation as a rigorous check of the learned Hamiltonian flow.

8. **Generalization and Performance Assessment**:
    - Assess trajectory reconstruction accuracy on held-out $N=50$ simulations.
    - Perform zero-shot transfer tests on $N=25$ and $N=100$ simulations to evaluate the robustness of the scale-invariant message passing.
    - Compare final MSE results against the baseline analytical point-mass gravity model.
# filename: codebase/step_2.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import torch
import torch.nn as nn

class GNNInteractionNetwork(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.epsilon = nn.Parameter(torch.tensor(0.01, dtype=torch.float32))
        self.edge_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        nn.init.zeros_(self.edge_mlp[-1].weight)
        nn.init.constant_(self.edge_mlp[-1].bias, 1.0)
        print("GNNInteractionNetwork instantiated with learnable epsilon initialized to: " + str(self.epsilon.item()))

    def edge_function(self, dx, r2):
        phys_force_mag = 1.0 / (r2 + self.epsilon**2)**1.5
        edge_inputs = r2.unsqueeze(-1)
        m_ij = self.edge_mlp(edge_inputs).squeeze(-1)
        force_mag = m_ij * phys_force_mag
        force_vectors = force_mag.unsqueeze(-1) * dx
        return force_vectors

    def forward(self, pos):
        batch_size, N, _ = pos.shape
        dx = pos.unsqueeze(1) - pos.unsqueeze(2)
        r2 = torch.sum(dx**2, dim=-1)
        mask = ~torch.eye(N, dtype=torch.bool, device=pos.device).unsqueeze(0)
        r2_safe = r2 + (~mask).float()
        force_vectors = self.edge_function(dx, r2_safe)
        force_vectors = force_vectors * mask.unsqueeze(-1)
        accelerations = torch.sum(force_vectors, dim=2)
        return accelerations

if __name__ == '__main__':
    model = GNNInteractionNetwork(hidden_dim=64)
    batch_size = 2
    N_particles = 5
    dummy_pos = torch.randn(batch_size, N_particles, 3)
    acc = model(dummy_pos)
    print("Input positions shape: " + str(dummy_pos.shape))
    print("Output accelerations shape: " + str(acc.shape))
    total_force = torch.sum(acc, dim=1)
    max_force_deviation = torch.max(torch.abs(total_force)).item()
    print("Maximum deviation from zero total force (Newton's Third Law check): " + str(max_force_deviation))
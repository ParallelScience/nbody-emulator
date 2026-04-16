# filename: codebase/step_2.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import torch
import torch.nn as nn

class InteractionNetwork(nn.Module):
    def __init__(self, hidden_dim=64, epsilon=0.01):
        super().__init__()
        self.epsilon = epsilon
        self.edge_mlp = nn.Sequential(
            nn.Linear(7, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3)
        )
        
    def phi(self, r_i, r_j, epsilon=0.01):
        dx = r_j - r_i
        d_sq = torch.sum(dx**2, dim=-1, keepdim=True) + epsilon**2
        f_phys = dx / (d_sq ** 1.5)
        edge_inputs = torch.cat([dx, f_phys, d_sq], dim=-1)
        return self.edge_mlp(edge_inputs)

    def forward(self, pos):
        has_batch = pos.dim() == 3
        if not has_batch:
            pos = pos.unsqueeze(0)
        B, N, _ = pos.shape
        pos_i = pos.unsqueeze(2)
        pos_j = pos.unsqueeze(1)
        h_ij = self.phi(pos_i, pos_j, self.epsilon)
        h_ji = h_ij.transpose(1, 2)
        F_ij = 0.5 * (h_ij - h_ji)
        eye = torch.eye(N, dtype=torch.bool, device=pos.device)
        F_ij = F_ij.masked_fill(eye.unsqueeze(0).unsqueeze(-1), 0.0)
        acc = torch.sum(F_ij, dim=2)
        if not has_batch:
            acc = acc.squeeze(0)
        return acc

def test_newtons_third_law():
    model = InteractionNetwork(hidden_dim=32, epsilon=0.01)
    model.eval()
    torch.manual_seed(42)
    pos = torch.randn(3, 3)
    acc = model(pos)
    total_force = torch.sum(acc, dim=0)
    print("--- Unit Test: Newton's Third Law ---")
    print("Particle Positions:\n" + str(pos.detach().numpy()))
    print("Predicted Accelerations (Forces):\n" + str(acc.detach().numpy()))
    print("Total Force (Sum of Accelerations):\n" + str(total_force.detach().numpy()))
    is_zero = torch.allclose(total_force, torch.zeros_like(total_force), atol=1e-6)
    if is_zero:
        print("SUCCESS: Newton's Third Law is verified. The sum of pairwise forces is numerically zero.")
    else:
        print("FAILURE: Newton's Third Law is violated!")
    assert is_zero, "Newton's Third Law violated!"

if __name__ == '__main__':
    test_newtons_third_law()
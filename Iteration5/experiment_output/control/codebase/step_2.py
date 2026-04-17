# filename: codebase/step_2.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import torch
import torch.nn as nn

class HNNPotential(nn.Module):
    def __init__(self, hidden_dim=64, eps=0.01):
        super().__init__()
        self.eps = eps
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )
        self._idx_cache = {}

    def forward(self, pos):
        batch_size, N, _ = pos.shape
        diff = pos.unsqueeze(2) - pos.unsqueeze(1)
        dist_sq = torch.sum(diff**2, dim=-1)
        dist_soft = torch.sqrt(dist_sq + self.eps**2)
        if N not in self._idx_cache or self._idx_cache[N][0].device != pos.device:
            self._idx_cache[N] = torch.triu_indices(N, N, offset=1, device=pos.device)
        idx_i, idx_j = self._idx_cache[N]
        d_ij = dist_soft[:, idx_i, idx_j]
        d_ij_flat = d_ij.reshape(-1, 1)
        u_ij = self.mlp(d_ij_flat)
        u_ij = u_ij.view(batch_size, -1)
        U = torch.sum(u_ij, dim=1, keepdim=True)
        U = U / N
        return U

def compute_force(pos, potential_model):
    with torch.enable_grad():
        if not pos.requires_grad:
            pos.requires_grad_(True)
        U = potential_model(pos)
        force = -torch.autograd.grad(
            outputs=U,
            inputs=pos,
            grad_outputs=torch.ones_like(U),
            create_graph=True
        )[0]
    return force

class NeuralLeapfrog(nn.Module):
    def __init__(self, potential_model, dt=0.01, steps=10):
        super().__init__()
        self.potential_model = potential_model
        self.dt = dt
        self.steps = steps

    def forward(self, pos, vel):
        if not pos.requires_grad:
            pos.requires_grad_(True)
        force = compute_force(pos, self.potential_model)
        for _ in range(self.steps):
            vel_half = vel + 0.5 * self.dt * force
            pos = pos + self.dt * vel_half
            force = compute_force(pos, self.potential_model)
            vel = vel_half + 0.5 * self.dt * force
        return pos, vel

if __name__ == '__main__':
    batch_size = 2
    N = 50
    pos = torch.randn(batch_size, N, 3)
    vel = torch.randn(batch_size, N, 3)
    model = HNNPotential(hidden_dim=32, eps=0.01)
    integrator = NeuralLeapfrog(model, dt=0.01, steps=5)
    pos_out, vel_out = integrator(pos, vel)
    loss = torch.sum(pos_out**2) + torch.sum(vel_out**2)
    loss.backward()
    for name, param in model.named_parameters():
        if param.grad is not None:
            print("Gradient computed for " + name + ", shape: " + str(param.grad.shape))
        else:
            print("No gradient for " + name)
    print("HNN architecture module successfully implemented and tested.")
# filename: codebase/step_2.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import torch
import torch.nn as nn

class EdgeNetwork(nn.Module):
    def __init__(self, hidden_dim=64, use_softening=True):
        super().__init__()
        self.use_softening = use_softening
        self.eps = 0.01
        self.net = nn.Sequential(nn.Linear(2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

    def forward(self, r_ij):
        dist_sq = torch.sum(r_ij**2, dim=-1, keepdim=True)
        if self.use_softening:
            dist_sq = dist_sq + self.eps**2
        else:
            dist_sq = dist_sq + 1e-8
        dist = torch.sqrt(dist_sq)
        inv_dist_cube = 1.0 / (dist_sq * dist)
        edge_features = torch.cat([dist, inv_dist_cube], dim=-1)
        m_ij = self.net(edge_features)
        f_ij = m_ij * r_ij
        return f_ij

class InteractionNetwork(nn.Module):
    def __init__(self, hidden_dim=64, use_softening=True):
        super().__init__()
        self.edge_net = EdgeNetwork(hidden_dim, use_softening)
        self._indices_cache = {}

    def forward(self, pos):
        batch_size, N, _ = pos.shape
        device = pos.device
        if (N, device) not in self._indices_cache:
            self._indices_cache[(N, device)] = torch.triu_indices(N, N, offset=1, device=device)
        i, j = self._indices_cache[(N, device)]
        r_ij = pos[:, j, :] - pos[:, i, :]
        f_ij = self.edge_net(r_ij)
        acc = torch.zeros_like(pos)
        acc.scatter_add_(1, i.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, 3), f_ij)
        acc.scatter_add_(1, j.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, 3), -f_ij)
        return acc

if __name__ == '__main__':
    batch_size = 2
    N = 50
    pos = torch.randn(batch_size, N, 3)
    model = InteractionNetwork(hidden_dim=64, use_softening=True)
    acc = model(pos)
    print('InteractionNetwork instantiated successfully.')
    print('Input position shape: ' + str(pos.shape))
    print('Output acceleration shape: ' + str(acc.shape))
    total_acc = acc.sum(dim=1)
    print('Total acceleration (should be ~0 due to Newton\'s Third Law):')
    print(str(total_acc.detach().numpy()))
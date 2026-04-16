# filename: codebase/step_2.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import numpy as np
import torch
import torch.nn as nn

class InteractionNetwork(nn.Module):
    def __init__(self, include_physics_prior=False, hidden_dim=64):
        super(InteractionNetwork, self).__init__()
        self.include_physics_prior = include_physics_prior
        in_dim = 9 if include_physics_prior else 6
        self.edge_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3)
        )
    def forward(self, pos, vel):
        batch_size, N, _ = pos.shape
        r_ij = pos.unsqueeze(1) - pos.unsqueeze(2)
        v_ij = vel.unsqueeze(1) - vel.unsqueeze(2)
        edge_features = [r_ij, v_ij]
        if self.include_physics_prior:
            eps = 0.01
            dist_sq = torch.sum(r_ij**2, dim=-1, keepdim=True) + eps**2
            f_phys = r_ij / (dist_sq**1.5)
            idx = torch.arange(N, device=pos.device)
            f_phys[:, idx, idx, :] = 0.0
            edge_features.append(f_phys)
        edge_inputs = torch.cat(edge_features, dim=-1)
        f_ij = self.edge_mlp(edge_inputs)
        f_ji = f_ij.transpose(1, 2)
        f_ij_sym = (f_ij - f_ji) / 2.0
        idx = torch.arange(N, device=pos.device)
        f_ij_sym[:, idx, idx, :] = 0.0
        acc = torch.sum(f_ij_sym, dim=2)
        return acc

def create_and_save_splits(n_sims=100, seed=42, data_dir='data/'):
    np.random.seed(seed)
    indices = np.random.permutation(n_sims)
    train_idx = indices[:80]
    val_idx = indices[80:90]
    test_idx = indices[90:]
    splits = {'train_idx': train_idx, 'val_idx': val_idx, 'test_idx': test_idx}
    save_path = os.path.join(data_dir, 'data_splits.npz')
    np.savez(save_path, **splits)
    print('Data splits successfully created and saved to ' + save_path)

if __name__ == '__main__':
    create_and_save_splits()
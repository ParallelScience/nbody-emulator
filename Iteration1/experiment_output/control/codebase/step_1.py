# filename: codebase/step_1.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import torch
from torch.utils.data import Dataset, DataLoader
import os

def compute_accelerations(pos, eps=0.01):
    dx = pos.unsqueeze(1) - pos.unsqueeze(2)
    r2 = torch.sum(dx**2, dim=-1)
    r3 = (r2 + eps**2)**1.5
    a = dx / r3.unsqueeze(-1)
    return torch.sum(a, dim=2)

def leapfrog_integrate(pos, vel, dt=0.01, steps=500, save_interval=50, eps=0.01):
    trajectories_pos = [pos.clone()]
    trajectories_vel = [vel.clone()]
    a = compute_accelerations(pos, eps)
    for step in range(1, steps + 1):
        vel_half = vel + 0.5 * dt * a
        pos = pos + dt * vel_half
        a = compute_accelerations(pos, eps)
        vel = vel_half + 0.5 * dt * a
        if step % save_interval == 0:
            trajectories_pos.append(pos.clone())
            trajectories_vel.append(vel.clone())
    return torch.stack(trajectories_pos, dim=1), torch.stack(trajectories_vel, dim=1)

class NBodyDataset(Dataset):
    def __init__(self, data_path="data/trajectories.pt", split="train", n_particles=50):
        data = torch.load(data_path, weights_only=True)
        self.pos = data['pos']
        self.vel = data['vel']
        if split == "train":
            self.pos = self.pos[:80]
            self.vel = self.vel[:80]
        elif split == "test":
            self.pos = self.pos[80:]
            self.vel = self.vel[80:]
        self.n_particles = n_particles
    def __len__(self):
        return len(self.pos)
    def __getitem__(self, idx):
        p = self.pos[idx, :, :self.n_particles, :]
        v = self.vel[idx, :, :self.n_particles, :]
        return p, v

if __name__ == '__main__':
    N_sim = 100
    N_part = 50
    pos_ic = torch.randn(N_sim, N_part, 3, dtype=torch.float64)
    vel_ic = torch.randn(N_sim, N_part, 3, dtype=torch.float64) * 0.1
    pos_traj, vel_traj = leapfrog_integrate(pos_ic, vel_ic, dt=0.01, steps=500, save_interval=50, eps=0.01)
    pos_traj = pos_traj.to(torch.float32)
    vel_traj = vel_traj.to(torch.float32)
    pos_rms = torch.sqrt(torch.mean(pos_ic.to(torch.float32)**2)).item()
    vel_rms = torch.sqrt(torch.mean(vel_ic.to(torch.float32)**2)).item()
    pos_traj_norm = pos_traj / pos_rms
    vel_traj_norm = vel_traj / vel_rms
    save_path = "data/trajectories.pt"
    torch.save({'pos': pos_traj_norm, 'vel': vel_traj_norm, 't': torch.linspace(0, 5.0, 11)}, save_path)
    dataset = NBodyDataset(data_path=save_path, split="train", n_particles=25)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    p_batch, v_batch = next(iter(loader))
    print("Dataset prepared and DataLoader tested.")
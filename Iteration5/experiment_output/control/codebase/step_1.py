# filename: codebase/step_1.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import numpy as np
import os

def compute_accelerations(pos, mass, eps=0.01):
    dx = pos[:, np.newaxis, :, :] - pos[:, :, np.newaxis, :]
    dist_sq = np.sum(dx**2, axis=-1) + eps**2
    dist_cube = dist_sq ** 1.5
    mass_j = mass[:, np.newaxis, :, np.newaxis]
    acc = np.sum(dx * mass_j / dist_cube[..., np.newaxis], axis=2)
    return acc

def generate_plummer_sphere(n_particles, n_sims):
    r = (np.random.rand(n_sims, n_particles) ** (-2/3) - 1) ** (-0.5)
    theta = np.arccos(2 * np.random.rand(n_sims, n_particles) - 1)
    phi = 2 * np.pi * np.random.rand(n_sims, n_particles)
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    pos = np.stack([x, y, z], axis=-1)
    vel = np.zeros_like(pos)
    mass = np.ones((n_sims, n_particles))
    return pos, vel, mass

if __name__ == '__main__':
    data_dir = 'data/'
    n_sims = 100
    n_particles = 50
    pos, vel, mass = generate_plummer_sphere(n_particles, n_sims)
    dt = 0.01
    steps = 500
    eps = 0.01
    snapshots = [np.concatenate([pos, vel], axis=-1)]
    acc = compute_accelerations(pos, mass, eps)
    for step in range(1, steps + 1):
        vel_half = vel + 0.5 * dt * acc
        pos = pos + dt * vel_half
        acc = compute_accelerations(pos, mass, eps)
        vel = vel_half + 0.5 * dt * acc
        if step % 50 == 0:
            snapshots.append(np.concatenate([pos, vel], axis=-1))
    snapshots = np.array(snapshots).transpose(1, 0, 2, 3)
    train_snapshots = snapshots[:80]
    val_snapshots = snapshots[80:]
    pos_mean = np.mean(train_snapshots[..., 0:3])
    pos_std = np.std(train_snapshots[..., 0:3])
    vel_mean = np.mean(train_snapshots[..., 3:6])
    vel_std = np.std(train_snapshots[..., 3:6])
    mean = np.array([pos_mean, pos_mean, pos_mean, vel_mean, vel_mean, vel_mean])
    std = np.array([pos_std, pos_std, pos_std, vel_std, vel_std, vel_std])
    train_snapshots_norm = (train_snapshots - mean) / (std + 1e-8)
    val_snapshots_norm = (val_snapshots - mean) / (std + 1e-8)
    np.savez(os.path.join(data_dir, 'normalization_stats.npz'), mean=mean, std=std)
    np.save(os.path.join(data_dir, 'train_trajectories.npy'), train_snapshots_norm)
    np.save(os.path.join(data_dir, 'val_trajectories.npy'), val_snapshots_norm)
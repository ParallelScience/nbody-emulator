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

def compute_energies(pos, vel, mass, eps=0.01):
    kin = 0.5 * np.sum(mass * np.sum(vel**2, axis=-1), axis=-1)
    dx = pos[:, np.newaxis, :, :] - pos[:, :, np.newaxis, :]
    dist = np.sqrt(np.sum(dx**2, axis=-1) + eps**2)
    mass_prod = mass[:, np.newaxis, :] * mass[:, :, np.newaxis]
    N = pos.shape[1]
    mask = np.triu(np.ones((N, N)), k=1)
    pot = -np.sum(mass_prod * mask / dist, axis=(1, 2))
    return kin, pot

if __name__ == '__main__':
    ic_final_path = 'data/ic_final.npy'
    metadata_path = 'data/sim_metadata.npy'
    data_dir = 'data/'
    data = np.load(ic_final_path)
    metadata = np.load(metadata_path)
    data = data.reshape(-1, 50, 13)
    ic_data = data[data[:, 0, 12] == 0]
    pos = ic_data[:, :, 0:3]
    vel = ic_data[:, :, 3:6]
    mass = ic_data[:, :, 6]
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
    snapshots = np.array(snapshots)
    snapshots = snapshots.transpose(1, 0, 2, 3)
    kin_0, pot_0 = compute_energies(snapshots[:, 0, :, 0:3], snapshots[:, 0, :, 3:6], mass, eps)
    kin_f, pot_f = compute_energies(snapshots[:, -1, :, 0:3], snapshots[:, -1, :, 3:6], mass, eps)
    E_0 = kin_0 + pot_0
    E_f = kin_f + pot_f
    rel_errors = np.abs(E_f - E_0) / np.abs(E_0)
    train_snapshots = snapshots[:80]
    val_snapshots = snapshots[80:]
    pos_mean = np.mean(train_snapshots[..., 0:3])
    pos_std = np.std(train_snapshots[..., 0:3])
    vel_mean = np.mean(train_snapshots[..., 3:6])
    vel_std = np.std(train_snapshots[..., 3:6])
    if pos_std == 0: pos_std = 1.0
    if vel_std == 0: vel_std = 1.0
    mean = np.array([pos_mean, pos_mean, pos_mean, vel_mean, vel_mean, vel_mean])
    std = np.array([pos_std, pos_std, pos_std, vel_std, vel_std, vel_std])
    train_snapshots_norm = (train_snapshots - mean) / std
    val_snapshots_norm = (val_snapshots - mean) / std
    norm_stats_path = os.path.join(data_dir, 'normalization_stats.npz')
    train_path = os.path.join(data_dir, 'train_trajectories.npy')
    val_path = os.path.join(data_dir, 'val_trajectories.npy')
    train_raw_path = os.path.join(data_dir, 'train_trajectories_raw.npy')
    val_raw_path = os.path.join(data_dir, 'val_trajectories_raw.npy')
    np.savez(norm_stats_path, mean=mean, std=std)
    np.save(train_path, train_snapshots_norm)
    np.save(val_path, val_snapshots_norm)
    np.save(train_raw_path, train_snapshots)
    np.save(val_raw_path, val_snapshots)
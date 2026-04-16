# filename: codebase/step_1.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import numpy as np
import os

def compute_acceleration_batch(pos, mass, epsilon=0.01):
    B, N, _ = pos.shape
    dx = pos[:, None, :, :] - pos[:, :, None, :]
    dist_sq = np.sum(dx**2, axis=-1) + epsilon**2
    dist_cube = dist_sq ** 1.5
    eye = np.eye(N, dtype=bool)[None, :, :]
    dist_cube[eye] = 1.0
    force_mag = mass[:, None, :] / dist_cube
    force_mag[eye] = 0.0
    acc = np.sum(force_mag[..., None] * dx, axis=2)
    return acc

def leapfrog_step(pos, vel, mass, dt, epsilon=0.01, acc=None):
    if acc is None:
        acc = compute_acceleration_batch(pos, mass, epsilon)
    vel_half = vel + 0.5 * dt * acc
    pos_next = pos + dt * vel_half
    acc_next = compute_acceleration_batch(pos_next, mass, epsilon)
    vel_next = vel_half + 0.5 * dt * acc_next
    return pos_next, vel_next, acc_next

def compute_energy(pos, vel, mass, epsilon=0.01):
    B, N, _ = pos.shape
    ke = 0.5 * np.sum(mass * np.sum(vel**2, axis=-1), axis=-1)
    dx = pos[:, None, :, :] - pos[:, :, None, :]
    dist_sq = np.sum(dx**2, axis=-1) + epsilon**2
    dist = np.sqrt(dist_sq)
    eye = np.eye(N, dtype=bool)[None, :, :]
    dist[eye] = 1.0
    pe_matrix = - (mass[:, :, None] * mass[:, None, :]) / dist
    pe_matrix[eye] = 0.0
    pe = 0.5 * np.sum(pe_matrix, axis=(1, 2))
    return ke, pe, ke + pe

def print_stats(name, data):
    print(name + " | Mean: " + str(np.mean(data)) + " | Std: " + str(np.std(data)) + " | Min: " + str(np.min(data)) + " | Max: " + str(np.max(data)))

if __name__ == '__main__':
    data_dir = "data/"
    ic_final_path = "/home/node/work/projects/nbody_emulator/data/ic_final.npy"
    sim_metadata_path = "/home/node/work/projects/nbody_emulator/data/sim_metadata.npy"
    ic_final = np.load(ic_final_path)
    sim_metadata = np.load(sim_metadata_path)
    if len(ic_final.shape) == 4:
        ic_final = ic_final.reshape(-1, ic_final.shape[2], ic_final.shape[3])
    time_per_snapshot = ic_final[:, 0, 12]
    unique_times = np.unique(time_per_snapshot)
    ic = ic_final[time_per_snapshot == np.min(unique_times)]
    final = ic_final[time_per_snapshot == np.max(unique_times)]
    dt = 0.01
    epsilon = 0.01
    steps = 500
    target_steps = [0, 50, 100, 150, 200, 250, 300, 350, 400, 450]
    pos = ic[:, :, 0:3].copy()
    vel = ic[:, :, 3:6].copy()
    mass = ic[:, :, 6].copy()
    pos_all = []
    vel_all = []
    acc_all = []
    delta_v_all = []
    time_all = []
    acc = compute_acceleration_batch(pos, mass, epsilon)
    ke_0, pe_0, te_0 = compute_energy(pos, vel, mass, epsilon)
    for step in range(steps + 1):
        if step in target_steps:
            pos_next, vel_next, acc_next = leapfrog_step(pos, vel, mass, dt, epsilon, acc)
            delta_v = vel_next - vel
            pos_all.append(pos.copy())
            vel_all.append(vel.copy())
            acc_all.append(acc.copy())
            delta_v_all.append(delta_v.copy())
            time_all.append(step * dt)
        if step < steps:
            pos, vel, acc = leapfrog_step(pos, vel, mass, dt, epsilon, acc)
    ke_end, pe_end, te_end = compute_energy(pos, vel, mass, epsilon)
    energy_error = np.abs((te_end - te_0) / te_0)
    pos_all = np.array(pos_all).transpose(1, 0, 2, 3)
    vel_all = np.array(vel_all).transpose(1, 0, 2, 3)
    acc_all = np.array(acc_all).transpose(1, 0, 2, 3)
    delta_v_all = np.array(delta_v_all).transpose(1, 0, 2, 3)
    time_all = np.array(time_all)
    mass_all = np.repeat(mass[:, None, :], len(target_steps), axis=1)
    pos_mean = np.mean(ic[:, :, 0:3], axis=(0, 1))
    pos_std = np.std(ic[:, :, 0:3])
    vel_mean = np.mean(ic[:, :, 3:6], axis=(0, 1))
    vel_std = np.std(ic[:, :, 3:6])
    acc_mean = np.mean(acc_all, axis=(0, 1, 2))
    acc_std = np.std(acc_all)
    delta_v_mean = np.mean(delta_v_all, axis=(0, 1, 2))
    delta_v_std = np.std(delta_v_all)
    pos_std = pos_std if pos_std != 0 else 1.0
    vel_std = vel_std if vel_std != 0 else 1.0
    acc_std = acc_std if acc_std != 0 else 1.0
    delta_v_std = delta_v_std if delta_v_std != 0 else 1.0
    pos_norm = (pos_all - pos_mean) / pos_std
    vel_norm = (vel_all - vel_mean) / vel_std
    acc_norm = (acc_all - acc_mean) / acc_std
    delta_v_norm = (delta_v_all - delta_v_mean) / delta_v_std
    stats = {'pos_mean': pos_mean, 'pos_std': pos_std, 'vel_mean': vel_mean, 'vel_std': vel_std, 'acc_mean': acc_mean, 'acc_std': acc_std, 'delta_v_mean': delta_v_mean, 'delta_v_std': delta_v_std}
    output_path = os.path.join(data_dir, "processed_trajectory_dataset.npz")
    np.savez(output_path, pos=pos_all, vel=vel_all, acc=acc_all, delta_v=delta_v_all, mass=mass_all, time=time_all, pos_norm=pos_norm, vel_norm=vel_norm, acc_norm=acc_norm, delta_v_norm=delta_v_norm, **stats)
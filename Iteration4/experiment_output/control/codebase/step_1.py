# filename: codebase/step_1.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import numpy as np
import os

def generate_plummer_sphere(n_particles, b=1.0):
    r = b / np.sqrt(np.random.rand(n_particles)**(-2/3) - 1)
    phi = np.random.rand(n_particles) * 2 * np.pi
    costheta = 2 * np.random.rand(n_particles) - 1
    theta = np.arccos(costheta)
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    pos = np.stack([x, y, z], axis=1)
    vel = np.zeros((n_particles, 3))
    return pos, vel

def get_accelerations(pos, eps=0.01):
    dx = pos[:, None, :, :] - pos[:, :, None, :]
    r2 = np.sum(dx**2, axis=-1) + eps**2
    r3 = r2**1.5
    a = np.sum(dx / r3[..., None], axis=2)
    return a

def compute_energies_vectorized(pos, vel, eps=0.01):
    ke = 0.5 * np.sum(vel**2, axis=(1, 2))
    dx = pos[:, None, :, :] - pos[:, :, None, :]
    r = np.sqrt(np.sum(dx**2, axis=-1) + eps**2)
    mask = ~np.eye(pos.shape[1], dtype=bool)
    pe = -0.5 * np.sum((1.0 / r) * mask, axis=(1, 2))
    return ke, pe

def save_partition(name, idx, data_dir, snapshots_pos, snapshots_vel, snapshots_acc, snapshots_t, initial_conditions, energy_initial):
    filepath = os.path.join(data_dir, name + ".npz")
    np.savez(filepath, pos=snapshots_pos[idx], vel=snapshots_vel[idx], acc=snapshots_acc[idx], t=snapshots_t, pos_init=initial_conditions[idx, :, 0:3], vel_init=initial_conditions[idx, :, 3:6], energy_init=energy_initial[idx])
    print("Saved " + name + " set to " + filepath + " with " + str(len(idx)) + " simulations.")

if __name__ == '__main__':
    N_sim = 100
    N_part = 50
    dt = 0.01
    n_steps = 500
    snapshot_interval = 50
    eps = 0.01
    pos_list = []
    vel_list = []
    for _ in range(N_sim):
        p, v = generate_plummer_sphere(N_part)
        pos_list.append(p)
        vel_list.append(v)
    pos = np.array(pos_list)
    vel = np.array(vel_list)
    initial_conditions = np.concatenate([pos, vel], axis=2)
    snapshots_pos, snapshots_vel, snapshots_acc, snapshots_t = [], [], [], []
    acc = get_accelerations(pos, eps)
    for step in range(1, n_steps + 1):
        v_half = vel + 0.5 * dt * acc
        pos = pos + dt * v_half
        acc = get_accelerations(pos, eps)
        vel = v_half + 0.5 * dt * acc
        if step % snapshot_interval == 0:
            snapshots_pos.append(pos.copy())
            snapshots_vel.append(vel.copy())
            snapshots_acc.append(acc.copy())
            snapshots_t.append(step * dt)
    snapshots_pos = np.array(snapshots_pos).transpose(1, 0, 2, 3)
    snapshots_vel = np.array(snapshots_vel).transpose(1, 0, 2, 3)
    snapshots_acc = np.array(snapshots_acc).transpose(1, 0, 2, 3)
    snapshots_t = np.array(snapshots_t)
    ke_initial, pe_initial = compute_energies_vectorized(initial_conditions[:, :, 0:3], initial_conditions[:, :, 3:6], eps)
    energy_initial = ke_initial + pe_initial
    indices = np.arange(N_sim)
    data_dir = "data/"
    save_partition("train_data", indices[:80], data_dir, snapshots_pos, snapshots_vel, snapshots_acc, snapshots_t, initial_conditions, energy_initial)
    save_partition("val_data", indices[80:90], data_dir, snapshots_pos, snapshots_vel, snapshots_acc, snapshots_t, initial_conditions, energy_initial)
    save_partition("test_data", indices[90:], data_dir, snapshots_pos, snapshots_vel, snapshots_acc, snapshots_t, initial_conditions, energy_initial)
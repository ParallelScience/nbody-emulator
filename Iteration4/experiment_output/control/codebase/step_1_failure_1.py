# filename: codebase/step_1.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import numpy as np
import os

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
    ic_final_path = '/home/node/work/projects/nbody_emulator/data/ic_final.npy'
    sim_metadata_path = '/home/node/work/projects/nbody_emulator/data/sim_metadata.npy'
    ic_final = np.load(ic_final_path)
    sim_metadata = np.load(sim_metadata_path)
    time_idx = 12
    flat_ic = ic_final.reshape(-1, 13)
    mask = flat_ic[:, time_idx] == 0
    initial_conditions = flat_ic[mask].reshape(-1, 50, 13)
    pos = initial_conditions[:, :, 0:3]
    vel = initial_conditions[:, :, 3:6]
    N_sim = pos.shape[0]
    dt = 0.01
    n_steps = 500
    snapshot_interval = 50
    eps = 0.01
    snapshots_pos = []
    snapshots_vel = []
    snapshots_acc = []
    snapshots_t = []
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
    ke_final, pe_final = compute_energies_vectorized(snapshots_pos[:, -1, :, :], snapshots_vel[:, -1, :, :], eps)
    energy_initial = ke_initial + pe_initial
    energy_final = ke_final + pe_final
    energy_error = np.abs((energy_final - energy_initial) / energy_initial)
    virial_ratio = 2 * ke_initial / np.abs(pe_initial)
    meta_energy_error = sim_metadata[:, 6]
    print("Mean Energy Error |(E_f - E_i)/E_i|: " + str(np.mean(energy_error)))
    indices = np.arange(N_sim)
    train_idx = indices[:80]
    val_idx = indices[80:90]
    test_idx = indices[90:]
    data_dir = "data/"
    save_partition("train_data", train_idx, data_dir, snapshots_pos, snapshots_vel, snapshots_acc, snapshots_t, initial_conditions, energy_initial)
    save_partition("val_data", val_idx, data_dir, snapshots_pos, snapshots_vel, snapshots_acc, snapshots_t, initial_conditions, energy_initial)
    save_partition("test_data", test_idx, data_dir, snapshots_pos, snapshots_vel, snapshots_acc, snapshots_t, initial_conditions, energy_initial)
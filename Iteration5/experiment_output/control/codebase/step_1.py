# filename: codebase/step_1.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime
import time

def get_accelerations(pos, eps=0.01):
    diff = pos[:, np.newaxis, :, :] - pos[:, :, np.newaxis, :]
    dist_sq = np.sum(diff**2, axis=-1) + eps**2
    inv_dist_cube = dist_sq**(-1.5)
    mask = np.eye(pos.shape[1], dtype=bool)
    inv_dist_cube[:, mask] = 0.0
    a = np.einsum('sijc,sij->sic', diff, inv_dist_cube)
    return a

def get_energies(pos, vel, eps=0.01):
    ke = 0.5 * np.sum(vel**2, axis=(1, 2))
    diff = pos[:, np.newaxis, :, :] - pos[:, :, np.newaxis, :]
    dist_sq = np.sum(diff**2, axis=-1) + eps**2
    inv_dist = dist_sq**(-0.5)
    mask = np.eye(pos.shape[1], dtype=bool)
    inv_dist[:, mask] = 0.0
    pe = -0.5 * np.sum(inv_dist, axis=(1, 2))
    return ke, pe, ke + pe

def leapfrog(pos, vel, dt=0.01, steps=500, save_interval=50):
    snapshots = [(pos.copy(), vel.copy())]
    a = get_accelerations(pos)
    for step in range(1, steps + 1):
        vel = vel + 0.5 * dt * a
        pos = pos + dt * vel
        a = get_accelerations(pos)
        vel = vel + 0.5 * dt * a
        if step % save_interval == 0:
            snapshots.append((pos.copy(), vel.copy()))
    return snapshots

def generate_initial_conditions(N_sim=100, N_part=50, b=1.0, eps=0.01):
    pos_init = np.zeros((N_sim, N_part, 3))
    vel_init = np.zeros((N_sim, N_part, 3))
    for i in range(N_sim):
        X1 = np.random.rand(N_part)
        r = b / np.sqrt(X1**(-2/3) - 1.0)
        phi = np.random.uniform(0, 2*np.pi, N_part)
        costheta = np.random.uniform(-1, 1, N_part)
        sintheta = np.sqrt(1 - costheta**2)
        pos = np.stack([r * sintheta * np.cos(phi), r * sintheta * np.sin(phi), r * costheta], axis=-1)
        pos -= np.mean(pos, axis=0)
        r_a = b / 5.0
        r_norm = np.linalg.norm(pos, axis=-1)
        v_esc = np.sqrt(2.0 * (1.0 + r_norm / r_a)**(-1))
        v_mag = v_esc * np.random.uniform(0, 1, N_part)
        phi_v = np.random.uniform(0, 2*np.pi, N_part)
        costheta_v = np.random.uniform(-1, 1, N_part)
        sintheta_v = np.sqrt(1 - costheta_v**2)
        vel = np.stack([v_mag * sintheta_v * np.cos(phi_v), v_mag * sintheta_v * np.sin(phi_v), v_mag * costheta_v], axis=-1)
        vel -= np.mean(vel, axis=0)
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=-1) + eps**2
        inv_dist = dist_sq**(-0.5)
        np.fill_diagonal(inv_dist, 0.0)
        pe = -0.5 * np.sum(inv_dist)
        ke = 0.5 * np.sum(vel**2)
        target_ke = 0.5 * np.abs(pe)
        vel *= np.sqrt(target_ke / ke)
        pos_init[i] = pos
        vel_init[i] = vel
    return pos_init, vel_init

if __name__ == '__main__':
    data_path = '/home/node/work/projects/nbody_emulator/data/ic_final.npy'
    meta_path = '/home/node/work/projects/nbody_emulator/data/sim_metadata.npy'
    try:
        data = np.load(data_path)
        meta = np.load(meta_path)
        if data.dtype.names is not None:
            ic_data = data[data['time'] == 0]
            pos_init = np.stack([ic_data['x'], ic_data['y'], ic_data['z']], axis=-1)
            vel_init = np.stack([ic_data['vx'], ic_data['vy'], ic_data['vz']], axis=-1)
            pos_init = pos_init.reshape(-1, 50, 3)
            vel_init = vel_init.reshape(-1, 50, 3)
        else:
            data_flat = data.reshape(-1, 50, 13)
            ic_data = data_flat[data_flat[:, 0, 12] == 0]
            if len(ic_data) == 0:
                ic_data = data_flat
            pos_init = ic_data[:, :, 0:3]
            vel_init = ic_data[:, :, 3:6]
    except FileNotFoundError:
        np.random.seed(42)
        pos_init, vel_init = generate_initial_conditions(N_sim=100, N_part=50, b=1.0, eps=0.01)
    snapshots = leapfrog(pos_init, vel_init, dt=0.01, steps=500, save_interval=50)
    pos_traj = np.stack([s[0] for s in snapshots], axis=1)
    vel_traj = np.stack([s[1] for s in snapshots], axis=1)
    ke_init, pe_init, e_init = get_energies(pos_traj[:, 0], vel_traj[:, 0])
    ke_final, pe_final, e_final = get_energies(pos_traj[:, -1], vel_traj[:, -1])
    energy_error = np.abs((e_final - e_init) / e_init)
    print('--- Summary Statistics ---')
    print('Positions - Mean: ' + str(np.mean(pos_traj)) + ', Std: ' + str(np.std(pos_traj)))
    print('Velocities - Mean: ' + str(np.mean(vel_traj)) + ', Std: ' + str(np.std(vel_traj)))
    print('Initial Total Energy - Mean: ' + str(np.mean(e_init)) + ', Std: ' + str(np.std(e_init)))
    print('Final Total Energy - Mean: ' + str(np.mean(e_final)) + ', Std: ' + str(np.std(e_final)))
    print('Energy Conservation Error - Mean: ' + str(np.mean(energy_error)) + ', Max: ' + str(np.max(energy_error)))
    timestamp = int(time.time())
    plot_filename = 'data/data_diagnostics_1_' + str(timestamp) + '.png'
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    axs[0].hist(e_init, bins=20, alpha=0.6, label='Initial Energy')
    axs[0].hist(e_final, bins=20, alpha=0.6, label='Final Energy')
    axs[0].set_xlabel('Total Energy (M L^2 / T^2)')
    axs[0].set_ylabel('Count')
    axs[0].set_title('Total Energy Distribution')
    axs[0].legend()
    axs[0].grid(True, alpha=0.3)
    r_init = np.linalg.norm(pos_traj[:, 0], axis=-1).flatten()
    r_final = np.linalg.norm(pos_traj[:, -1], axis=-1).flatten()
    axs[1].hist(r_init, bins=40, alpha=0.6, label='Initial Radii', density=True)
    axs[1].hist(r_final, bins=40, alpha=0.6, label='Final Radii', density=True)
    axs[1].set_xlabel('Radius (L)')
    axs[1].set_ylabel('Density')
    axs[1].set_title('Particle Radii Distribution')
    axs[1].legend()
    axs[1].grid(True, alpha=0.3)
    v_init = np.linalg.norm(vel_traj[:, 0], axis=-1).flatten()
    v_final = np.linalg.norm(vel_traj[:, -1], axis=-1).flatten()
    axs[2].hist(v_init, bins=40, alpha=0.6, label='Initial Speeds', density=True)
    axs[2].hist(v_final, bins=40, alpha=0.6, label='Final Speeds', density=True)
    axs[2].set_xlabel('Speed (L / T)')
    axs[2].set_ylabel('Density')
    axs[2].set_title('Particle Speeds Distribution')
    axs[2].legend()
    axs[2].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_filename, dpi=300)
    plt.close()
    pos_mean = np.mean(pos_traj)
    pos_std = np.std(pos_traj)
    vel_mean = np.mean(vel_traj)
    vel_std = np.std(vel_traj)
    pos_norm = (pos_traj - pos_mean) / pos_std
    vel_norm = (vel_traj - vel_mean) / vel_std
    traj_norm = np.concatenate([pos_norm, vel_norm], axis=-1)
    train_data = traj_norm[:80]
    val_data = traj_norm[80:]
    np.save('data/train_dataset.npy', train_data)
    np.save('data/val_dataset.npy', val_data)
    np.savez('data/norm_stats.npz', pos_mean=pos_mean, pos_std=pos_std, vel_mean=vel_mean, vel_std=vel_std)
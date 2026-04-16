# filename: codebase/step_1.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import numpy as np
import os
import matplotlib.pyplot as plt
import datetime
import time

def compute_acceleration_batch(pos, eps=0.01):
    diff = pos[:, :, np.newaxis, :] - pos[:, np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=-1)
    dist_sq += eps**2
    factor = 1.0 / (dist_sq**1.5)
    idx = np.arange(pos.shape[1])
    factor[:, idx, idx] = 0.0
    a = -np.sum(diff * factor[:, :, :, np.newaxis], axis=2)
    return a

if __name__ == '__main__':
    ic_path = '/home/node/work/projects/nbody_emulator/data/ic_final.npy'
    meta_path = '/home/node/work/projects/nbody_emulator/data/sim_metadata.npy'
    data = np.load(ic_path)
    metadata = np.load(meta_path)
    if len(data.shape) == 3:
        times = data[:, 0, 12]
        ic_data = data[times == 0]
        if len(ic_data) == 0:
            ic_data = data[times < 1e-5]
    elif len(data.shape) == 4:
        times = data[:, :, 0, 12]
        ic_data = []
        for i in range(data.shape[0]):
            if times[i, 0] == 0:
                ic_data.append(data[i, 0])
            else:
                ic_data.append(data[i, 1])
        ic_data = np.array(ic_data)
    else:
        raise ValueError('Unexpected data shape: ' + str(data.shape))
    pos_ic = ic_data[:, :, 0:3]
    vel_ic = ic_data[:, :, 3:6]
    dt = 0.01
    n_steps = 500
    eps = 0.01
    snapshot_steps = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
    snapshots = []
    pos = pos_ic.copy()
    vel = vel_ic.copy()
    a_current = compute_acceleration_batch(pos, eps)
    for step in range(1, n_steps + 1):
        v_half = vel + 0.5 * dt * a_current
        pos = pos + dt * v_half
        a_current = compute_acceleration_batch(pos, eps)
        vel = v_half + 0.5 * dt * a_current
        if step in snapshot_steps:
            snapshots.append({'time': step * dt, 'pos': pos.copy(), 'vel': vel.copy(), 'acc': a_current.copy()})
    pos_snapshots = np.array([s['pos'] for s in snapshots])
    vel_snapshots = np.array([s['vel'] for s in snapshots])
    acc_snapshots = np.array([s['acc'] for s in snapshots])
    times_arr = np.array([s['time'] for s in snapshots])
    np.savez('data/intermediate_snapshots.npz', time=times_arr, pos=pos_snapshots, vel=vel_snapshots, acc=acc_snapshots)
    dt_lyap = 0.001
    n_steps_lyap = 5000
    pos_ref = pos_ic.copy()
    vel_ref = vel_ic.copy()
    delta_r0 = 1e-8
    np.random.seed(42)
    pert = np.random.randn(*pos_ref.shape)
    norm_pert = np.linalg.norm(pert.reshape(pert.shape[0], -1), axis=1)
    pert = pert / norm_pert[:, np.newaxis, np.newaxis] * delta_r0
    pos_pert = pos_ref + pert
    vel_pert = vel_ic.copy()
    a_ref = compute_acceleration_batch(pos_ref, eps)
    a_pert = compute_acceleration_batch(pos_pert, eps)
    times_lyap = []
    distances = []
    for step in range(1, n_steps_lyap + 1):
        v_half_ref = vel_ref + 0.5 * dt_lyap * a_ref
        pos_ref = pos_ref + dt_lyap * v_half_ref
        a_ref = compute_acceleration_batch(pos_ref, eps)
        vel_ref = v_half_ref + 0.5 * dt_lyap * a_ref
        v_half_pert = vel_pert + 0.5 * dt_lyap * a_pert
        pos_pert = pos_pert + dt_lyap * v_half_pert
        a_pert = compute_acceleration_batch(pos_pert, eps)
        vel_pert = v_half_pert + 0.5 * dt_lyap * a_pert
        if step % 100 == 0:
            t = step * dt_lyap
            dist = np.linalg.norm((pos_ref - pos_pert).reshape(pos_ref.shape[0], -1), axis=1)
            times_lyap.append(t)
            distances.append(dist)
    times_lyap = np.array(times_lyap)
    distances = np.array(distances)
    distances = np.maximum(distances, 1e-15)
    log_dist = np.log(distances)
    lambdas = []
    for i in range(log_dist.shape[1]):
        valid_idx = distances[:, i] < 0.1
        if np.sum(valid_idx) > 10:
            p = np.polyfit(times_lyap[valid_idx], log_dist[valid_idx, i], 1)
            lambdas.append(p[0])
        else:
            p = np.polyfit(times_lyap, log_dist[:, i], 1)
            lambdas.append(p[0])
    lambdas = np.array(lambdas)
    valid_lambdas = lambdas[lambdas > 0]
    lyapunov_times = 1.0 / valid_lambdas
    plt.rcParams['text.usetex'] = False
    plt.figure(figsize=(8, 6))
    plt.hist(lyapunov_times, bins=20, color='skyblue', edgecolor='black')
    plt.xlabel('Lyapunov Time t_lambda (Normalized Units)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Global Lyapunov Times across 100 Simulations')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    plot_filename = 'data/lyapunov_histogram_1_' + timestamp + '.png'
    plt.savefig(plot_filename, dpi=300)
    print('Histogram saved to ' + plot_filename)
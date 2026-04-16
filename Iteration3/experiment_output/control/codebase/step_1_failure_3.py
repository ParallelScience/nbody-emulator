# filename: codebase/step_1.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import numpy as np
import matplotlib.pyplot as plt
import datetime
import time
import os

def generate_plummer_sphere(n_particles, n_sims=100):
    pos = np.random.randn(n_sims, n_particles, 3)
    r = np.linalg.norm(pos, axis=-1)
    pos = pos / r[..., np.newaxis] * (1 + r**2)**(-0.25)[..., np.newaxis]
    vel = np.random.randn(n_sims, n_particles, 3) * 0.1
    return pos, vel

def compute_acceleration_batch(pos, eps=0.01):
    diff = pos[:, :, np.newaxis, :] - pos[:, np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=-1) + eps**2
    factor = 1.0 / (dist_sq**1.5)
    idx = np.arange(pos.shape[1])
    factor[:, idx, idx] = 0.0
    a = -np.sum(diff * factor[:, :, :, np.newaxis], axis=2)
    return a

if __name__ == '__main__':
    n_particles = 50
    n_sims = 100
    pos_ic, vel_ic = generate_plummer_sphere(n_particles, n_sims)
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
    pert = np.random.randn(*pos_ref.shape) * delta_r0
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
    distances = np.maximum(np.array(distances), 1e-15)
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
    lyapunov_times = 1.0 / np.array(lambdas)
    plt.figure(figsize=(8, 6))
    plt.hist(lyapunov_times[lyapunov_times > 0], bins=20, color='skyblue', edgecolor='black')
    plt.xlabel('Lyapunov Time t_lambda')
    plt.ylabel('Frequency')
    plt.title('Distribution of Global Lyapunov Times')
    plt.tight_layout()
    plot_filename = 'data/lyapunov_histogram.png'
    plt.savefig(plot_filename, dpi=300)
    print('Histogram saved to ' + plot_filename)
# filename: codebase/step_1.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import numpy as np
import os

def generate_plummer_sphere(N, b=1.0):
    X = np.random.uniform(0.001, 0.999, N)
    r = b / np.sqrt(X**(-2/3) - 1.0)
    phi = np.random.uniform(0, 2*np.pi, N)
    costheta = np.random.uniform(-1, 1, N)
    sintheta = np.sqrt(1 - costheta**2)
    pos = np.zeros((N, 3))
    pos[:, 0] = r * sintheta * np.cos(phi)
    pos[:, 1] = r * sintheta * np.sin(phi)
    pos[:, 2] = r * costheta
    vel = np.zeros((N, 3))
    for i in range(N):
        x = r[i] / b
        v_esc = np.sqrt(2.0 / np.sqrt(x**2 + 1.0))
        while True:
            q = np.random.uniform(0, 1)
            g = q**2 * (1.0 - q**2)**3.5
            if np.random.uniform(0, 0.1) < g:
                v_mag = q * v_esc
                break
        v_phi = np.random.uniform(0, 2*np.pi)
        v_costheta = np.random.uniform(-1, 1)
        v_sintheta = np.sqrt(1 - v_costheta**2)
        vel[i, 0] = v_mag * v_sintheta * np.cos(v_phi)
        vel[i, 1] = v_mag * v_sintheta * np.sin(v_phi)
        vel[i, 2] = v_mag * v_costheta
    pos -= np.mean(pos, axis=0)
    vel -= np.mean(vel, axis=0)
    return pos, vel

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

def leapfrog_step(pos, vel, mass, dt, epsilon=0.01, acc=None):
    if acc is None:
        acc = compute_acceleration_batch(pos, mass, epsilon)
    vel_half = vel + 0.5 * dt * acc
    pos_next = pos + dt * vel_half
    acc_next = compute_acceleration_batch(pos_next, mass, epsilon)
    vel_next = vel_half + 0.5 * dt * acc_next
    return pos_next, vel_next, acc_next

def print_stats(name, data):
    print(name + " | Mean: " + str(np.mean(data)) + " | Std: " + str(np.std(data)) + " | Min: " + str(np.min(data)) + " | Max: " + str(np.max(data)))

if __name__ == '__main__':
    data_dir = "data/"
    num_sims = 100
    N_particles = 50
    epsilon = 0.01
    dt = 0.01
    steps = 500
    np.random.seed(42)
    pos_ic = np.zeros((num_sims, N_particles, 3))
    vel_ic = np.zeros((num_sims, N_particles, 3))
    mass_ic = np.ones((num_sims, N_particles))
    for i in range(num_sims):
        p, v = generate_plummer_sphere(N_particles, b=1.0)
        pos_ic[i] = p
        vel_ic[i] = v
    ke_initial, pe_initial, _ = compute_energy(pos_ic, vel_ic, mass_ic, epsilon)
    for i in range(num_sims):
        q = np.sqrt(0.5 * np.abs(pe_initial[i]) / ke_initial[i])
        vel_ic[i] *= q
    ke_0, pe_0, te_0 = compute_energy(pos_ic, vel_ic, mass_ic, epsilon)
    target_steps = [0, 50, 100, 150, 200, 250, 300, 350, 400, 450]
    pos = pos_ic.copy()
    vel = vel_ic.copy()
    mass = mass_ic.copy()
    pos_all, vel_all, acc_all, delta_v_all, time_all = [], [], [], [], []
    acc = compute_acceleration_batch(pos, mass, epsilon)
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
    pos_all = np.array(pos_all).transpose(1, 0, 2, 3)
    vel_all = np.array(vel_all).transpose(1, 0, 2, 3)
    acc_all = np.array(acc_all).transpose(1, 0, 2, 3)
    delta_v_all = np.array(delta_v_all).transpose(1, 0, 2, 3)
    time_all = np.array(time_all)
    mass_all = np.repeat(mass[:, None, :], len(target_steps), axis=1)
    pos_mean, pos_std = np.mean(pos_ic, axis=(0, 1)), np.std(pos_ic)
    vel_mean, vel_std = np.mean(vel_ic, axis=(0, 1)), np.std(vel_ic)
    acc_mean, acc_std = np.mean(acc_all, axis=(0, 1, 2)), np.std(acc_all)
    delta_v_mean, delta_v_std = np.mean(delta_v_all, axis=(0, 1, 2)), np.std(delta_v_all)
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
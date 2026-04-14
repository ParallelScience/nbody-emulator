"""Generate 100 N-body simulations for the nbody_emulator project."""

import numpy as np
import os

np.random.seed(42)

N_SIMS = 100
N_PARTICLES = 50
DT = 0.01
N_STEPS = 500
T = float(N_STEPS * DT)
G = 1.0
M = 1.0
B_PLUMMER = 1.0


def plummer_positions(n, b=1.0):
    u = np.random.rand(n)
    r = b * u ** (1/3) / ((1 - u) ** (2/3))
    cos_theta = 2 * np.random.rand(n) - 1
    phi = 2 * np.pi * np.random.rand(n)
    sin_theta = np.sqrt(1 - cos_theta**2)
    x = r * sin_theta * np.cos(phi)
    y = r * sin_theta * np.sin(phi)
    z = r * cos_theta
    return np.stack([x, y, z], axis=1)


def plummer_velocities(pos, b=1.0, r_a=None):
    if r_a is None:
        r_a = b / 5.0
    r_norm = np.linalg.norm(pos, axis=1)
    v_esc_sq = 2.0 * (1.0 + r_norm / r_a) ** (-1.0)
    v_mag = np.sqrt(-v_esc_sq * np.log(1.0 - np.random.rand(len(r_norm))))
    cos_theta = 2.0 * np.random.rand(len(r_norm)) - 1.0
    phi = 2.0 * np.pi * np.random.rand(len(r_norm))
    sin_theta = np.sqrt(1.0 - cos_theta**2)
    vx = v_mag * sin_theta * np.cos(phi)
    vy = v_mag * sin_theta * np.sin(phi)
    vz = v_mag * cos_theta
    return np.stack([vx, vy, vz], axis=1)


def compute_acceleration(pos, m=1.0, G=1.0, eps2=0.01):
    """Vectorized gravitational acceleration. O(N^2)."""
    n = pos.shape[0]
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]  # (n, n, 3)
    r2 = np.sum(diff**2, axis=2) + eps2  # (n, n)
    r3 = r2 * np.sqrt(r2)  # r^3
    # a_i = sum_j G*m*(r_j - r_i)/|...|^3 = -sum_j G*m*diff[i,j]/r3[i,j]
    F = -G * m / r3  # (n, n)
    np.fill_diagonal(F, 0.0)  # exclude self-interaction
    acc = np.sum(F[:, :, np.newaxis] * diff, axis=1)  # (n, 3)
    return acc


def kinetic_energy_total(v, m=1.0):
    return 0.5 * m * np.sum(v**2)


def kinetic_energy_per_particle(v, m=1.0):
    return 0.5 * m * np.sum(v**2, axis=1)


def potential_energy_total(pos, m=1.0, G=1.0, eps=0.01):
    """Pairwise potential energy (scalar), excluding self-terms via masked array."""
    n = pos.shape[0]
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r = np.sqrt(np.sum(diff**2, axis=2) + eps)
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    r_masked = np.ma.array(r, mask=~mask)
    return -G * m * m * np.ma.sum(1.0 / r_masked)


def leapfrog_step(pos, vel, dt, m=1.0, G=1.0, eps2=0.01):
    acc = compute_acceleration(pos, m, G, eps2)
    vel_half = vel + 0.5 * dt * acc
    pos_new = pos + dt * vel_half
    acc_new = compute_acceleration(pos_new, m, G, eps2)
    vel_new = vel_half + 0.5 * dt * acc_new
    return pos_new, vel_new


print("Generating 100 N-body simulations...")
ic_data = np.zeros((N_SIMS, N_PARTICLES, 13), dtype=np.float64)
final_data = np.zeros((N_SIMS, N_PARTICLES, 13), dtype=np.float64)
metadata = np.zeros((N_SIMS, 7), dtype=np.float64)

for sim_id in range(N_SIMS):
    if sim_id % 20 == 0:
        print(f"  Simulation {sim_id}/100")
    
    pos = plummer_positions(N_PARTICLES, B_PLUMMER)
    vel = plummer_velocities(pos, B_PLUMMER)
    
    # Virialize
    pe0 = potential_energy_total(pos, M, G)
    ke0 = kinetic_energy_total(vel, M)
    target_ke = 0.5 * abs(pe0)
    vel *= np.sqrt(target_ke / ke0)
    
    # Center of mass
    pos -= np.mean(pos, axis=0)
    vel -= np.mean(vel, axis=0)
    
    ke0, pe0 = kinetic_energy_total(vel, M), potential_energy_total(pos, M, G)
    
    pos_cur, vel_cur = pos.copy(), vel.copy()
    for step in range(N_STEPS):
        pos_cur, vel_cur = leapfrog_step(pos_cur, vel_cur, DT, M, G)
    
    keT, peT = kinetic_energy_total(vel_cur, M), potential_energy_total(pos_cur, M, G)
    
    r_ic = np.linalg.norm(pos, axis=1)
    v_ic = np.linalg.norm(vel, axis=1)
    r_fin = np.linalg.norm(pos_cur, axis=1)
    v_fin = np.linalg.norm(vel_cur, axis=1)
    ke_part_ic = kinetic_energy_per_particle(vel, M)
    ke_part_fin = kinetic_energy_per_particle(vel_cur, M)
    
    # ic: x,y,z,vx,vy,vz,mass,KE,PE,totalE,r,v,time
    ic_data[sim_id, :, 0] = pos[:, 0]
    ic_data[sim_id, :, 1] = pos[:, 1]
    ic_data[sim_id, :, 2] = pos[:, 2]
    ic_data[sim_id, :, 3] = vel[:, 0]
    ic_data[sim_id, :, 4] = vel[:, 1]
    ic_data[sim_id, :, 5] = vel[:, 2]
    ic_data[sim_id, :, 6] = M
    ic_data[sim_id, :, 7] = ke_part_ic
    ic_data[sim_id, :, 8] = pe0 / N_PARTICLES
    ic_data[sim_id, :, 9] = ke_part_ic + pe0 / N_PARTICLES
    ic_data[sim_id, :, 10] = r_ic
    ic_data[sim_id, :, 11] = v_ic
    ic_data[sim_id, :, 12] = 0.0
    
    # final
    final_data[sim_id, :, 0] = pos_cur[:, 0]
    final_data[sim_id, :, 1] = pos_cur[:, 1]
    final_data[sim_id, :, 2] = pos_cur[:, 2]
    final_data[sim_id, :, 3] = vel_cur[:, 0]
    final_data[sim_id, :, 4] = vel_cur[:, 1]
    final_data[sim_id, :, 5] = vel_cur[:, 2]
    final_data[sim_id, :, 6] = M
    final_data[sim_id, :, 7] = ke_part_fin
    final_data[sim_id, :, 8] = peT / N_PARTICLES
    final_data[sim_id, :, 9] = ke_part_fin + peT / N_PARTICLES
    final_data[sim_id, :, 10] = r_fin
    final_data[sim_id, :, 11] = v_fin
    final_data[sim_id, :, 12] = T
    
    E_initial = ke0 + pe0
    err = abs(keT + peT - E_initial) / abs(E_initial) if E_initial != 0 else 0.0
    metadata[sim_id] = [sim_id, N_PARTICLES, ke0, pe0, keT, peT, err]

data_dir = '/home/node/work/projects/nbody_emulator/data'
os.makedirs(data_dir, exist_ok=True)

np.save(f'{data_dir}/ic.npy', ic_data)
np.save(f'{data_dir}/final.npy', final_data)
np.save(f'{data_dir}/metadata.npy', metadata)

print(f"\nSaved:")
print(f"  ic.npy:       {ic_data.shape}")
print(f"  final.npy:    {final_data.shape}")
print(f"  metadata.npy: {metadata.shape}")

errs = metadata[:, 6]
print(f"\nEnergy error (|dE/E|):")
print(f"  Mean: {np.nanmean(errs):.2e}")
print(f"  Max:  {np.nanmax(errs):.2e}")
print(f"  Min:  {np.nanmin(errs):.2e}")
print("Done!")

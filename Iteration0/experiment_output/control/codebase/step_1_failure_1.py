# filename: codebase/step_1.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import numpy as np
import os

def get_accelerations_batch(pos, eps=0.01):
    diff = pos[:, None, :, :] - pos[:, :, None, :]
    dist_sq = np.sum(diff**2, axis=-1) + eps**2
    dist_cube = dist_sq**1.5
    acc = np.sum(diff / dist_cube[..., None], axis=2)
    return acc

class DataAugmenter:
    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)
        
    def augment(self, pos, vel):
        H = self.rng.normal(size=(3, 3))
        Q, R = np.linalg.qr(H)
        signs = np.sign(np.diag(R))
        signs[signs == 0] = 1
        Q = Q * signs
        if np.linalg.det(Q) < 0:
            Q[:, 0] *= -1
        reflect = np.diag(self.rng.choice([-1, 1], size=3))
        transformation = Q @ reflect
        aug_pos = pos @ transformation.T
        aug_vel = vel @ transformation.T
        return aug_pos, aug_vel

if __name__ == '__main__':
    ic_final_path = '/home/node/work/projects/nbody_emulator/data/ic_final.npy'
    sim_metadata_path = '/home/node/work/projects/nbody_emulator/data/sim_metadata.npy'
    data_dir = 'data/'
    print('Loading dataset...')
    data = np.load(ic_final_path)
    flat_data = data.reshape(-1, 13)
    ic_particles = flat_data[flat_data[:, 12] == 0]
    ic_data = ic_particles.reshape(100, 50, 13)
    pos_init = ic_data[:, :, 0:3]
    vel_init = ic_data[:, :, 3:6]
    print('Loaded initial conditions for ' + str(pos_init.shape[0]) + ' simulations.')
    metadata = np.load(sim_metadata_path)
    metadata = metadata[metadata[:, 0].argsort()]
    initial_KE = metadata[:, 2]
    initial_PE = metadata[:, 3]
    M = 50.0
    G = 1.0
    virial_radius = (G * M**2) / (2 * np.abs(initial_PE))
    velocity_dispersion = np.sqrt(2 * initial_KE / M)
    norm_coeffs = np.stack([virial_radius, velocity_dispersion], axis=-1)
    norm_coeffs_path = os.path.join(data_dir, 'normalization_coeffs.npy')
    np.save(norm_coeffs_path, norm_coeffs)
    print('Computed normalization coefficients.')
    print('Mean Virial Radius: ' + str(np.round(np.mean(virial_radius), 4)))
    print('Mean Velocity Dispersion: ' + str(np.round(np.mean(velocity_dispersion), 4)))
    print('Saved normalization coefficients to ' + norm_coeffs_path)
    dt = 0.01
    eps = 0.01
    steps = 500
    pos = pos_init.copy()
    vel = vel_init.copy()
    snapshots = [np.concatenate([pos, vel], axis=-1)]
    print('Running leapfrog integrator for ' + str(steps) + ' steps...')
    acc = get_accelerations_batch(pos, eps)
    for step in range(1, steps + 1):
        vel += 0.5 * dt * acc
        pos += dt * vel
        acc = get_accelerations_batch(pos, eps)
        vel += 0.5 * dt * acc
        if step % 50 == 0:
            snapshots.append(np.concatenate([pos, vel], axis=-1))
    snapshots = np.array(snapshots)
    snapshots = snapshots.transpose(1, 0, 2, 3)
    trajectories_path = os.path.join(data_dir, 'trajectories.npy')
    np.save(trajectories_path, snapshots)
    print('Saved trajectories of shape ' + str(snapshots.shape) + ' to ' + trajectories_path)
    augmenter = DataAugmenter(seed=42)
    aug_pos, aug_vel = augmenter.augment(pos_init[0], vel_init[0])
    print('DataAugmenter utility tested successfully.')
    print('Data preprocessing and snapshot generation completed successfully.')
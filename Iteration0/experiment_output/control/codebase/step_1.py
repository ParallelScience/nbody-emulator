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
    ic_path = '/home/node/work/projects/nbody_emulator/data/ic.npy'
    metadata_path = '/home/node/work/projects/nbody_emulator/data/metadata.npy'
    data_dir = 'data/'
    print('Loading dataset...')
    pos_init = np.load(ic_path)[:, :, 0:3]
    vel_init = np.load(ic_path)[:, :, 3:6]
    metadata = np.load(metadata_path)
    metadata = metadata[metadata[:, 0].argsort()]
    initial_KE = metadata[:, 2]
    initial_PE = metadata[:, 3]
    M = 50.0
    G = 1.0
    virial_radius = (G * M**2) / (2 * np.abs(initial_PE))
    velocity_dispersion = np.sqrt(2 * initial_KE / M)
    norm_coeffs = np.stack([virial_radius, velocity_dispersion], axis=-1)
    np.save(os.path.join(data_dir, 'normalization_coeffs.npy'), norm_coeffs)
    dt = 0.01
    eps = 0.01
    steps = 500
    pos = pos_init.copy()
    vel = vel_init.copy()
    snapshots = [np.concatenate([pos, vel], axis=-1)]
    acc = get_accelerations_batch(pos, eps)
    for step in range(1, steps + 1):
        vel += 0.5 * dt * acc
        pos += dt * vel
        acc = get_accelerations_batch(pos, eps)
        vel += 0.5 * dt * acc
        if step % 50 == 0:
            snapshots.append(np.concatenate([pos, vel], axis=-1))
    snapshots = np.array(snapshots).transpose(1, 0, 2, 3)
    np.save(os.path.join(data_dir, 'trajectories.npy'), snapshots)
    print('Data preprocessing and snapshot generation completed successfully.')
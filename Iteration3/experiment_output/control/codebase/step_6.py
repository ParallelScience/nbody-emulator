# filename: codebase/step_6.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from step_1 import generate_plummer_sphere, compute_acceleration_batch
from step_2 import InteractionNetwork
from step_5 import compute_energy

def generate_test_data(n_particles, n_sims=10):
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
    pos_snapshots = np.array([s['pos'] for s in snapshots]).transpose(1, 0, 2, 3)
    vel_snapshots = np.array([s['vel'] for s in snapshots]).transpose(1, 0, 2, 3)
    acc_snapshots = np.array([s['acc'] for s in snapshots]).transpose(1, 0, 2, 3)
    return pos_ic, vel_ic, pos_snapshots, vel_snapshots, acc_snapshots

def evaluate_mse(model, pos_snapshots, vel_snapshots, acc_snapshots):
    device = torch.device('cpu')
    model.eval()
    criterion = nn.MSELoss()
    pos_flat = pos_snapshots.reshape(-1, pos_snapshots.shape[2], 3)
    vel_flat = vel_snapshots.reshape(-1, vel_snapshots.shape[2], 3)
    acc_flat = acc_snapshots.reshape(-1, acc_snapshots.shape[2], 3)
    dataset = TensorDataset(torch.tensor(pos_flat, dtype=torch.float32), torch.tensor(vel_flat, dtype=torch.float32), torch.tensor(acc_flat, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    mse = 0.0
    with torch.no_grad():
        for p, v, a in loader:
            p, v, a = p.to(device), v.to(device), a.to(device)
            pred_a = model(p, v)
            loss = criterion(pred_a, a)
            mse += loss.item() * p.size(0)
    mse /= len(loader.dataset)
    return mse

def rollout_energy_drift(model, pos_0, vel_0, dt=0.01, n_steps=500):
    pos = torch.tensor(pos_0, dtype=torch.float32).unsqueeze(0)
    vel = torch.tensor(vel_0, dtype=torch.float32).unsqueeze(0)
    energies = [compute_energy(pos_0, vel_0)]
    with torch.no_grad():
        acc = model(pos, vel)
        for step in range(1, n_steps + 1):
            v_half = vel + 0.5 * dt * acc
            pos = pos + dt * v_half
            acc = model(pos, v_half)
            vel = v_half + 0.5 * dt * acc
            if step % 10 == 0:
                e = compute_energy(pos[0].cpu().numpy(), vel[0].cpu().numpy())
                energies.append(e)
    energies = np.array(energies)
    e0 = energies[0]
    return np.var(energies / e0)

def main():
    device = torch.device('cpu')
    model = InteractionNetwork(include_physics_prior=True, hidden_dim=64).to(device)
    model.load_state_dict(torch.load('data/gnn_physics_informed.pth', map_location=device))
    model.eval()
    results = {}
    data = np.load('data/intermediate_snapshots.npz')
    pos = data['pos'].transpose(1, 0, 2, 3)
    vel = data['vel'].transpose(1, 0, 2, 3)
    acc = data['acc'].transpose(1, 0, 2, 3)
    splits = np.load('data/data_splits.npz')
    test_idx = splits['test_idx']
    pos_ic_50, vel_ic_50 = generate_plummer_sphere(50, 100)
    mse_50 = evaluate_mse(model, pos[test_idx], vel[test_idx], acc[test_idx])
    drifts_50 = [rollout_energy_drift(model, pos_ic_50[idx], vel_ic_50[idx]) for idx in test_idx]
    results[50] = {'mse': mse_50, 'drift': np.mean(drifts_50)}
    for n in [25, 100]:
        pos_ic, vel_ic, pos_snap, vel_snap, acc_snap = generate_test_data(n, 10)
        mse = evaluate_mse(model, pos_snap, vel_snap, acc_snap)
        drifts = [rollout_energy_drift(model, pos_ic[i], vel_ic[i]) for i in range(10)]
        results[n] = {'mse': mse, 'drift': np.mean(drifts)}
    print("=" * 60)
    print(f"{'N-Scaling Generalization Test Results':^60}")
    print("=" * 60)
    print(f"{'N Particles':<15} | {'Acceleration MSE':<20} | {'Energy Drift (Var(E/E0))':<20}")
    print("-" * 60)
    for n in [25, 50, 100]:
        print(f"{n:<15} | {results[n]['mse']:<20.6f} | {results[n]['drift']:<20.6e}")
    print("=" * 60)

if __name__ == '__main__':
    main()
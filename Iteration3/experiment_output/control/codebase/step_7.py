# filename: codebase/step_7.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from step_1 import generate_plummer_sphere, compute_acceleration_batch
from step_2 import InteractionNetwork
class DirectMLP(nn.Module):
    def __init__(self, in_dim=300, hidden_dim=1024, out_dim=300):
        super(DirectMLP, self).__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, out_dim))
    def forward(self, x):
        return self.net(x)
def compute_energy_batch(pos, vel, eps=0.01):
    ke = 0.5 * np.sum(vel**2, axis=(1, 2))
    diff = pos[:, :, np.newaxis, :] - pos[:, np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=-1) + eps**2
    idx = np.triu_indices(pos.shape[1], k=1)
    pe = -np.sum(1.0 / np.sqrt(dist_sq[:, idx[0], idx[1]]), axis=1)
    return ke + pe
def compute_lyapunov(pos_ref, vel_ref):
    dt_lyap = 0.001
    n_steps_lyap = 5000
    eps = 0.01
    delta_r0 = 1e-8
    np.random.seed(42)
    pert = np.random.randn(*pos_ref.shape)
    norm_pert = np.linalg.norm(pert.reshape(pert.shape[0], -1), axis=1)
    pert = pert / norm_pert[:, np.newaxis, np.newaxis] * delta_r0
    pos_pert = pos_ref + pert
    vel_pert = vel_ref.copy()
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
    return np.mean(lyapunov_times)
if __name__ == '__main__':
    pos_ic, vel_ic = generate_plummer_sphere(50, 100)
    data = np.load('data/intermediate_snapshots.npz')
    pos_snaps = data['pos']
    vel_snaps = data['vel']
    splits = np.load('data/data_splits.npz')
    train_idx = splits['train_idx']
    val_idx = splits['val_idx']
    test_idx = splits['test_idx']
    X = np.concatenate([pos_ic.reshape(100, -1), vel_ic.reshape(100, -1)], axis=1)
    Y = np.concatenate([pos_snaps[-1].reshape(100, -1), vel_snaps[-1].reshape(100, -1)], axis=1)
    X_train, Y_train = X[train_idx], Y[train_idx]
    X_val, Y_val = X[val_idx], Y[val_idx]
    X_test, Y_test = X[test_idx], Y[test_idx]
    device = torch.device('cpu')
    model_mlp = DirectMLP().to(device)
    optimizer = optim.Adam(model_mlp.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=50, factor=0.5)
    criterion = nn.MSELoss()
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(Y_train, dtype=torch.float32))
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_x = torch.tensor(X_val, dtype=torch.float32).to(device)
    val_y = torch.tensor(Y_val, dtype=torch.float32).to(device)
    for epoch in range(100):
        model_mlp.train()
        for x_b, y_b in train_loader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            pred = model_mlp(x_b)
            loss = criterion(pred, y_b)
            loss.backward()
            optimizer.step()
        model_mlp.eval()
        with torch.no_grad():
            val_pred = model_mlp(val_x)
            val_loss = criterion(val_pred, val_y).item()
        scheduler.step(val_loss)
    model_mlp.eval()
    with torch.no_grad():
        Y_pred_mlp = model_mlp(torch.tensor(X_test, dtype=torch.float32)).numpy()
    mse_mlp_t5 = np.mean((Y_pred_mlp - Y_test)**2)
    print('Direct MLP MSE at t=5.0: ' + str(mse_mlp_t5))
    lyap_time = compute_lyapunov(pos_ic[test_idx], vel_ic[test_idx])
    print('Mean Lyapunov time: ' + str(lyap_time))
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.title('MSE vs Time')
    plt.xlabel('Time')
    plt.ylabel('MSE')
    plt.subplot(1, 3, 2)
    plt.title('Energy Drift')
    plt.xlabel('Time')
    plt.ylabel('Var(E/E0)')
    plt.subplot(1, 3, 3)
    plt.title('Trajectory Divergence')
    plt.xlabel('Time / Lyapunov Time')
    plt.ylabel('Divergence')
    plt.tight_layout()
    plt.savefig('data/benchmark_results.png')
    print('Saved to data/benchmark_results.png')
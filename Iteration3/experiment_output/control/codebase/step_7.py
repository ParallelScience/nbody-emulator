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
from step_5 import compute_energy
class MLPDirect(nn.Module):
    def __init__(self, in_dim=300, hidden_dim=512, out_dim=300):
        super(MLPDirect, self).__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, out_dim))
    def forward(self, x):
        return self.net(x)
def train_mlp_direct(train_x, train_y):
    device = torch.device('cpu')
    model = MLPDirect().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    dataset = TensorDataset(torch.tensor(train_x, dtype=torch.float32), torch.tensor(train_y, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    epochs = 500
    for epoch in range(epochs):
        model.train()
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
    return model
def rollout_gnn(model, pos_0, vel_0, dt=0.01, n_steps=500, snapshot_steps=None):
    if snapshot_steps is None:
        snapshot_steps = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
    pos = torch.tensor(pos_0, dtype=torch.float32)
    vel = torch.tensor(vel_0, dtype=torch.float32)
    pos_preds = []
    vel_preds = []
    with torch.no_grad():
        acc = model(pos, vel)
        for step in range(1, n_steps + 1):
            v_half = vel + 0.5 * dt * acc
            pos = pos + dt * v_half
            acc = model(pos, v_half)
            vel = v_half + 0.5 * dt * acc
            if step in snapshot_steps:
                pos_preds.append(pos.cpu().numpy())
                vel_preds.append(vel.cpu().numpy())
    return np.array(pos_preds).transpose(1, 0, 2, 3), np.array(vel_preds).transpose(1, 0, 2, 3)
def get_energies(pos_array, vel_array):
    energies = []
    for i in range(pos_array.shape[0]):
        energies.append(compute_energy(pos_array[i], vel_array[i]))
    return np.array(energies)
def compute_lyapunov(pos_ic, vel_ic, eps=0.01):
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
    return np.mean(lyapunov_times[lyapunov_times > 0])
def main():
    pos_ic, vel_ic = generate_plummer_sphere(50, 100)
    data = np.load('data/intermediate_snapshots.npz')
    pos_snaps = data['pos'].transpose(1, 0, 2, 3)
    vel_snaps = data['vel'].transpose(1, 0, 2, 3)
    splits = np.load('data/data_splits.npz')
    train_idx = splits['train_idx']
    test_idx = splits['test_idx']
    X = np.concatenate([pos_ic.reshape(100, -1), vel_ic.reshape(100, -1)], axis=1)
    Y = np.concatenate([pos_snaps[:, -1].reshape(100, -1), vel_snaps[:, -1].reshape(100, -1)], axis=1)
    train_x, train_y = X[train_idx], Y[train_idx]
    test_x, test_y = X[test_idx], Y[test_idx]
    mlp_model = train_mlp_direct(train_x, train_y)
    mlp_model.eval()
    with torch.no_grad():
        test_pred_y = mlp_model(torch.tensor(test_x, dtype=torch.float32)).numpy()
    pos_pred_mlp = test_pred_y[:, :150].reshape(-1, 50, 3)
    vel_pred_mlp = test_pred_y[:, 150:].reshape(-1, 50, 3)
    device = torch.device('cpu')
    gnn_model = InteractionNetwork(include_physics_prior=True, hidden_dim=64).to(device)
    gnn_model.load_state_dict(torch.load('data/gnn_physics_informed.pth', map_location=device))
    gnn_model.eval()
    pos_preds_gnn, vel_preds_gnn = rollout_gnn(gnn_model, pos_ic[test_idx], vel_ic[test_idx])
    times = np.linspace(0.5, 5.0, 10)
    mse_gnn_time = np.mean((pos_preds_gnn - pos_snaps[test_idx])**2 + (vel_preds_gnn - vel_snaps[test_idx])**2, axis=(0, 2, 3))
    mse_mlp_final = np.mean((pos_pred_mlp - pos_snaps[test_idx, -1])**2 + (vel_pred_mlp - vel_snaps[test_idx, -1])**2)
    E_0 = get_energies(pos_ic[test_idx], vel_ic[test_idx])
    E_gnn_time = []
    for t_idx in range(10):
        E_t = get_energies(pos_preds_gnn[:, t_idx], vel_preds_gnn[:, t_idx])
        E_gnn_time.append(np.mean(np.abs(E_t - E_0) / np.abs(E_0)))
    E_gnn_time = np.array(E_gnn_time)
    E_mlp_final = get_energies(pos_pred_mlp, vel_pred_mlp)
    E_mlp_err = np.mean(np.abs(E_mlp_final - E_0) / np.abs(E_0))
    E_true_time = []
    for t_idx in range(10):
        E_t = get_energies(pos_snaps[test_idx, t_idx], vel_snaps[test_idx, t_idx])
        E_true_time.append(np.mean(np.abs(E_t - E_0) / np.abs(E_0)))
    E_true_time = np.array(E_true_time)
    div_gnn_time = np.mean(np.linalg.norm(pos_preds_gnn - pos_snaps[test_idx], axis=-1), axis=(0, 2))
    mean_lyap_time = compute_lyapunov(pos_ic[test_idx], vel_ic[test_idx])
    plt.rcParams['text.usetex'] = False
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    axs[0].plot(times, mse_gnn_time, marker='o', label='GNN Emulator')
    axs[0].scatter([5.0], [mse_mlp_final], color='red', zorder=5, label='Baseline MLP (Direct)')
    axs[0].set_yscale('log')
    axs[0].set_xlabel('Integration Time (T)')
    axs[0].set_ylabel('Phase-Space MSE')
    axs[0].set_title('(a) MSE vs Integration Time')
    axs[0].legend()
    axs[0].grid(True, alpha=0.5)
    axs[1].plot(times, E_gnn_time, marker='o', label='GNN Emulator')
    axs[1].plot(times, E_true_time, marker='x', linestyle='--', color='gray', label='Ground Truth (Leapfrog)')
    axs[1].scatter([5.0], [E_mlp_err], color='red', zorder=5, label='Baseline MLP (Direct)')
    axs[1].set_yscale('log')
    axs[1].set_xlabel('Integration Time (T)')
    axs[1].set_ylabel('Relative Energy Error |E - E_0| / |E_0|')
    axs[1].set_title('(b) Energy Conservation')
    axs[1].legend()
    axs[1].grid(True, alpha=0.5)
    axs[2].plot(times, div_gnn_time, marker='o', label='GNN Divergence')
    axs[2].axvline(x=mean_lyap_time, color='purple', linestyle='--', label='Mean Lyapunov Time (' + str(round(mean_lyap_time, 2)) + ')')
    axs[2].set_yscale('log')
    axs[2].set_xlabel('Integration Time (T)')
    axs[2].set_ylabel('Mean Particle Distance (L)')
    axs[2].set_title('(c) Trajectory Divergence')
    axs[2].legend()
    axs[2].grid(True, alpha=0.5)
    plt.tight_layout()
    timestamp = str(int(time.time()))
    plot_filename = os.path.join('data', 'benchmark_validation_1_' + timestamp + '.png')
    plt.savefig(plot_filename, dpi=300)
    print('Composite figure saved to ' + plot_filename)
    print('\n' + '='*60)
    print('BENCHMARKING AND PERFORMANCE VALIDATION RESULTS')
    print('='*60)
    print('Mean Lyapunov Time: ' + str(round(mean_lyap_time, 4)) + ' T')
    print('-' * 60)
    print('Phase-Space MSE at Final State (t=5.0):')
    print('  GNN Emulator:       ' + str(round(mse_gnn_time[-1], 6)))
    print('  Baseline MLP:       ' + str(round(mse_mlp_final, 6)))
    print('-' * 60)
    print('Relative Energy Error |E - E_0| / |E_0| at Final State (t=5.0):')
    print('  GNN Emulator:       ' + str(E_gnn_time[-1]))
    print('  Baseline MLP:       ' + str(E_mlp_err))
    print('  Ground Truth (LF):  ' + str(E_true_time[-1]))
    print('-' * 60)
    print('Trajectory Divergence (Mean Particle Distance) at Final State (t=5.0):')
    print('  GNN Emulator:       ' + str(round(div_gnn_time[-1], 6)))
    print('='*60 + '\n')
if __name__ == '__main__':
    main()
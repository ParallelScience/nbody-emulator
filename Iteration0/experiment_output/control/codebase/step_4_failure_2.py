# filename: codebase/step_4.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time
try:
    from step_2 import InteractionNetwork
except ImportError:
    InteractionNetwork = None
class BaselineMLP(nn.Module):
    def __init__(self, input_dim=300, hidden_dim=512, output_dim=300):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, output_dim))
    def forward(self, x):
        return self.net(x)
class SymplecticNeuralODE(nn.Module):
    def __init__(self, hidden_dim=64, use_softening=True):
        super().__init__()
        if InteractionNetwork is not None:
            self.gnn = InteractionNetwork(hidden_dim, use_softening)
        else:
            self.gnn = None
    def forward(self, pos, vel, t_span, dt=0.01):
        snapshots = []
        current_time = 0.0
        p = pos.clone()
        v = vel.clone()
        acc = self.gnn(p)
        for target_time in t_span:
            while current_time < target_time - 1e-5:
                step_dt = min(dt, target_time - current_time)
                v = v + 0.5 * step_dt * acc
                p = p + step_dt * v
                acc = self.gnn(p)
                v = v + 0.5 * step_dt * acc
                current_time += step_dt
            snapshots.append(torch.cat([p, v], dim=-1))
        return torch.stack(snapshots, dim=1)
def compute_energy(pos, vel, eps=0.01):
    ke = 0.5 * torch.sum(vel**2, dim=(1, 2))
    diff = pos.unsqueeze(2) - pos.unsqueeze(1)
    dist_sq = torch.sum(diff**2, dim=-1) + eps**2
    dist = torch.sqrt(dist_sq)
    inv_dist = 1.0 / dist
    batch_size, N, _ = pos.shape
    mask = torch.eye(N, dtype=torch.bool, device=pos.device).unsqueeze(0)
    inv_dist.masked_fill_(mask, 0.0)
    pe = -0.5 * torch.sum(inv_dist, dim=(1, 2))
    return ke + pe
def normalize(data, norm):
    pos = data[:, :, :3] / norm[:, 0, None, None]
    vel = data[:, :, 3:] / norm[:, 1, None, None]
    return np.concatenate([pos, vel], axis=-1)
def denormalize(data, norm):
    pos = data[:, :, :3] * norm[:, 0, None, None]
    vel = data[:, :, 3:] * norm[:, 1, None, None]
    return np.concatenate([pos, vel], axis=-1)
if __name__ == '__main__':
    data_dir = 'data/'
    trajectories = np.load(os.path.join(data_dir, 'trajectories.npy'))
    norm_coeffs = np.load(os.path.join(data_dir, 'normalization_coeffs.npy'))
    train_traj = trajectories[:80]
    test_traj = trajectories[80:]
    train_norm = norm_coeffs[:80]
    test_norm = norm_coeffs[80:]
    X_train = train_traj[:, 0, :, :]
    Y_train = train_traj[:, 10, :, :]
    X_test = test_traj[:, 0, :, :]
    Y_test = test_traj[:, 10, :, :]
    X_train_norm = normalize(X_train, train_norm).reshape(80, -1)
    Y_train_norm = normalize(Y_train, train_norm).reshape(80, -1)
    X_test_norm = normalize(X_test, test_norm).reshape(20, -1)
    Y_test_norm = normalize(Y_test, test_norm).reshape(20, -1)
    X_train_t = torch.tensor(X_train_norm, dtype=torch.float32)
    Y_train_t = torch.tensor(Y_train_norm, dtype=torch.float32)
    X_test_t = torch.tensor(X_test_norm, dtype=torch.float32)
    Y_test_t = torch.tensor(Y_test_norm, dtype=torch.float32)
    mlp = BaselineMLP()
    optimizer = optim.Adam(mlp.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    epochs = 500
    for epoch in range(epochs):
        optimizer.zero_grad()
        pred = mlp(X_train_t)
        loss = criterion(pred, Y_train_t)
        loss.backward()
        optimizer.step()
    torch.save(mlp.state_dict(), os.path.join(data_dir, 'mlp_model.pth'))
    mlp.eval()
    with torch.no_grad():
        Y_pred_norm = mlp(X_test_t).numpy().reshape(20, 50, 6)
    Y_pred = denormalize(Y_pred_norm, test_norm)
    mlp_pos_mse = np.mean((Y_pred[:, :, :3] - Y_test[:, :, :3])**2)
    mlp_vel_mse = np.mean((Y_pred[:, :, 3:] - Y_test[:, :, 3:])**2)
    node_path = os.path.join(data_dir, 'node_model.pth')
    node_available = False
    if os.path.exists(node_path):
        node = SymplecticNeuralODE(hidden_dim=64, use_softening=True)
        try:
            node.load_state_dict(torch.load(node_path))
            node.eval()
            node_available = True
        except Exception:
            pass
    if not node_available:
        node = SymplecticNeuralODE(hidden_dim=64, use_softening=True)
        optimizer_node = optim.Adam(node.parameters(), lr=1e-3)
        t_span = torch.linspace(0.5, 5.0, 10)
        epochs_node = 50
        train_traj_norm = np.zeros_like(train_traj)
        for i in range(11):
            train_traj_norm[:, i, :, :] = normalize(train_traj[:, i, :, :], train_norm)
        for epoch in range(epochs_node):
            optimizer_node.zero_grad()
            pos_in = torch.tensor(train_traj_norm[:, 0, :, :3], dtype=torch.float32)
            vel_in = torch.tensor(train_traj_norm[:, 0, :, 3:], dtype=torch.float32)
            pred_traj_norm = node(pos_in, vel_in, t_span, dt=0.05)
            target_traj_norm = torch.tensor(train_traj_norm[:, 1:, :, :], dtype=torch.float32)
            loss_mse = nn.MSELoss()(pred_traj_norm, target_traj_norm)
            loss_mse.backward()
            torch.nn.utils.clip_grad_norm_(node.parameters(), max_norm=1.0)
            optimizer_node.step()
        torch.save(node.state_dict(), node_path)
        node.eval()
        node_available = True
    if node_available:
        with torch.no_grad():
            X_test_norm_3d = normalize(X_test, test_norm)
            pos_in_test = torch.tensor(X_test_norm_3d[:, :, :3], dtype=torch.float32)
            vel_in_test = torch.tensor(X_test_norm_3d[:, :, 3:], dtype=torch.float32)
            t_span_test = torch.linspace(0.5, 5.0, 10)
            node_pred_traj_norm = node(pos_in_test, vel_in_test, t_span_test, dt=0.01)
            node_pred_traj = np.zeros((20, 10, 50, 6))
            for i in range(10):
                node_pred_traj[:, i, :, :] = denormalize(node_pred_traj_norm[:, i, :, :].numpy(), test_norm)
            Y_pred_node = node_pred_traj[:, -1, :, :]
        node_pos_mse = np.mean((Y_pred_node[:, :, :3] - Y_test[:, :, :3])**2)
        node_vel_mse = np.mean((Y_pred_node[:, :, 3:] - Y_test[:, :, 3:])**2)
        e_init = compute_energy(torch.tensor(X_test[:, :, :3]), torch.tensor(X_test[:, :, 3:])).numpy()
        e_true_final = compute_energy(torch.tensor(Y_test[:, :, :3]), torch.tensor(Y_test[:, :, 3:])).numpy()
        e_mlp_final = compute_energy(torch.tensor(Y_pred[:, :, :3]), torch.tensor(Y_pred[:, :, 3:])).numpy()
        e_node_final = compute_energy(torch.tensor(Y_pred_node[:, :, :3]), torch.tensor(Y_pred_node[:, :, 3:])).numpy()
        mlp_energy_error = np.abs((e_mlp_final - e_init) / e_init)
        node_energy_error = np.abs((e_node_final - e_init) / e_init)
        true_energy_error = np.abs((e_true_final - e_init) / e_init)
        print('\n--- Final Evaluation Metrics (Test Set: 20 simulations) ---')
        print(f'MLP Position MSE: {mlp_pos_mse:.6f}')
        print(f'MLP Velocity MSE: {mlp_vel_mse:.6f}')
        print(f'MLP Final Energy Error: {np.mean(mlp_energy_error):.6e}')
        print('')
        print(f'Neural ODE Position MSE: {node_pos_mse:.6f}')
        print(f'Neural ODE Velocity MSE: {node_vel_mse:.6f}')
        print(f'Neural ODE Final Energy Error: {np.mean(node_energy_error):.6e}')
        print('')
        print(f'Ground Truth Final Energy Error (Leapfrog): {np.mean(true_energy_error):.6e}')
        print('-----------------------------------------------------------\n')
        node_energies = []
        true_energies = []
        node_energies.append(e_init)
        true_energies.append(e_init)
        for i in range(10):
            e_node = compute_energy(torch.tensor(node_pred_traj[:, i, :, :3]), torch.tensor(node_pred_traj[:, i, :, 3:])).numpy()
            e_true = compute_energy(torch.tensor(test_traj[:, i+1, :, :3]), torch.tensor(test_traj[:, i+1, :, 3:])).numpy()
            node_energies.append(e_node)
            true_energies.append(e_true)
        node_energies = np.array(node_energies)
        true_energies = np.array(true_energies)
        node_energy_drift = np.mean(np.abs((node_energies - e_init) / e_init), axis=1)
        true_energy_drift = np.mean(np.abs((true_energies - e_init) / e_init), axis=1)
        mlp_times = [0, 5.0]
        mlp_energy_drift = [0, np.mean(mlp_energy_error)]
        times = np.linspace(0, 5.0, 11)
        plt.figure(figsize=(8, 6))
        plt.plot(times, true_energy_drift, label='Ground Truth (Leapfrog)', marker='o')
        plt.plot(times, node_energy_drift, label='Symplectic Neural ODE', marker='s')
        plt.plot(mlp_times, mlp_energy_drift, label='Baseline MLP', marker='^', linestyle='--')
        plt.xlabel('Time (T)')
        plt.ylabel('Relative Energy Error |(E(t) - E(0))/E(0)|')
        plt.yscale('log')
        plt.title('Energy Drift over Predicted Trajectory')
        plt.legend()
        plt.grid(True, which='both', ls='--')
        plt.tight_layout()
        timestamp = int(time.time())
        plot_path = os.path.join(data_dir, f'energy_drift_1_{timestamp}.png')
        plt.savefig(plot_path, dpi=300)
        print(f'Plot saved to {plot_path}')
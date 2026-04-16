# filename: codebase/step_5.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from step_1 import compute_acceleration_batch, compute_energy
from step_2 import InteractionNetwork
plt.rcParams['text.usetex'] = False
class BaselineMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(300, 512), nn.ReLU(), nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, 300))
    def forward(self, x):
        return self.net(x)
def main():
    data_dir = 'data/'
    dataset_path = os.path.join(data_dir, 'processed_trajectory_dataset.npz')
    dataset = np.load(dataset_path)
    pos_ic = dataset['pos'][:, 0, :, :]
    vel_ic = dataset['vel'][:, 0, :, :]
    mass = dataset['mass'][:, 0, :]
    epsilon = 0.01
    dt = 0.01
    steps = 500
    pos_gt = pos_ic.copy()
    vel_gt = vel_ic.copy()
    acc_gt = compute_acceleration_batch(pos_gt, mass, epsilon)
    for step in range(steps):
        vel_half = vel_gt + 0.5 * dt * acc_gt
        pos_gt = pos_gt + dt * vel_half
        acc_gt = compute_acceleration_batch(pos_gt, mass, epsilon)
        vel_gt = vel_half + 0.5 * dt * acc_gt
    pos_final_gt = pos_gt
    vel_final_gt = vel_gt
    X = np.concatenate([pos_ic.reshape(100, 150), vel_ic.reshape(100, 150)], axis=1)
    Y = np.concatenate([pos_final_gt.reshape(100, 150), vel_final_gt.reshape(100, 150)], axis=1)
    X_train = torch.tensor(X[:80], dtype=torch.float32)
    Y_train = torch.tensor(Y[:80], dtype=torch.float32)
    X_val = torch.tensor(X[80:], dtype=torch.float32)
    X_mean = X_train.mean(dim=0, keepdim=True)
    X_std = X_train.std(dim=0, keepdim=True) + 1e-6
    Y_mean = Y_train.mean(dim=0, keepdim=True)
    Y_std = Y_train.std(dim=0, keepdim=True) + 1e-6
    X_train_norm = (X_train - X_mean) / X_std
    Y_train_norm = (Y_train - Y_mean) / Y_std
    X_val_norm = (X_val - X_mean) / X_std
    device = torch.device('cpu')
    mlp = BaselineMLP().to(device)
    optimizer = optim.Adam(mlp.parameters(), lr=0.001)
    X_train_norm = X_train_norm.to(device)
    Y_train_norm = Y_train_norm.to(device)
    for epoch in range(3000):
        mlp.train()
        optimizer.zero_grad()
        pred = mlp(X_train_norm)
        loss = nn.functional.mse_loss(pred, Y_train_norm)
        loss.backward()
        optimizer.step()
    mlp.eval()
    with torch.no_grad():
        pred_Y_val_norm = mlp(X_val_norm.to(device)).cpu()
        pred_Y_val = pred_Y_val_norm * Y_std + Y_mean
    pred_pos_mlp = pred_Y_val[:, :150].reshape(20, 50, 3).numpy()
    pred_vel_mlp = pred_Y_val[:, 150:].reshape(20, 50, 3).numpy()
    gnn = InteractionNetwork(hidden_dim=64, epsilon=0.01).to(device)
    model_path = os.path.join(data_dir, 'gnn_model.pth')
    gnn.load_state_dict(torch.load(model_path, map_location=device))
    gnn.eval()
    pos_gnn = torch.tensor(pos_ic[80:], dtype=torch.float32).to(device)
    vel_gnn = torch.tensor(vel_ic[80:], dtype=torch.float32).to(device)
    mass_t = torch.tensor(mass[80:], dtype=torch.float32).to(device)
    with torch.no_grad():
        acc_gnn = gnn(pos_gnn)
        for step in range(steps):
            vel_half = vel_gnn + 0.5 * dt * acc_gnn
            pos_gnn = pos_gnn + dt * vel_half
            acc_gnn = gnn(pos_gnn)
            vel_gnn = vel_half + 0.5 * dt * acc_gnn
    pred_pos_gnn = pos_gnn.cpu().numpy()
    pred_vel_gnn = vel_gnn.cpu().numpy()
    pos_val_gt = pos_final_gt[80:]
    vel_val_gt = vel_final_gt[80:]
    mse_pos_mlp = np.mean((pred_pos_mlp - pos_val_gt)**2, axis=-1).flatten()
    mse_vel_mlp = np.mean((pred_vel_mlp - vel_val_gt)**2, axis=-1).flatten()
    mse_pos_gnn = np.mean((pred_pos_gnn - pos_val_gt)**2, axis=-1).flatten()
    mse_vel_gnn = np.mean((pred_vel_gnn - vel_val_gt)**2, axis=-1).flatten()
    mse_pos_mlp = np.nan_to_num(mse_pos_mlp, nan=1e10, posinf=1e10, neginf=1e10)
    mse_vel_mlp = np.nan_to_num(mse_vel_mlp, nan=1e10, posinf=1e10, neginf=1e10)
    mse_pos_gnn = np.nan_to_num(mse_pos_gnn, nan=1e10, posinf=1e10, neginf=1e10)
    mse_vel_gnn = np.nan_to_num(mse_vel_gnn, nan=1e10, posinf=1e10, neginf=1e10)
    _, _, e_initial = compute_energy(pos_ic[80:], vel_ic[80:], mass[80:], epsilon)
    _, _, e_final_gt = compute_energy(pos_val_gt, vel_val_gt, mass[80:], epsilon)
    _, _, e_final_mlp = compute_energy(pred_pos_mlp, pred_vel_mlp, mass[80:], epsilon)
    _, _, e_final_gnn = compute_energy(pred_pos_gnn, pred_vel_gnn, mass[80:], epsilon)
    err_e_gt = np.abs((e_final_gt - e_initial) / e_initial)
    err_e_mlp = np.abs((e_final_mlp - e_initial) / e_initial)
    err_e_gnn = np.abs((e_final_gnn - e_initial) / e_initial)
    err_e_gt = np.nan_to_num(err_e_gt, nan=1e10, posinf=1e10, neginf=1e10)
    err_e_mlp = np.nan_to_num(err_e_mlp, nan=1e10, posinf=1e10, neginf=1e10)
    err_e_gnn = np.nan_to_num(err_e_gnn, nan=1e10, posinf=1e10, neginf=1e10)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    mse_pos_mlp_safe = np.clip(mse_pos_mlp, 1e-10, None)
    mse_pos_gnn_safe = np.clip(mse_pos_gnn, 1e-10, None)
    bins_pos = np.logspace(np.log10(min(np.min(mse_pos_mlp_safe), np.min(mse_pos_gnn_safe))), np.log10(max(np.max(mse_pos_mlp_safe), np.max(mse_pos_gnn_safe))), 50)
    axes[0].hist(mse_pos_mlp_safe, bins=bins_pos, alpha=0.5, label='MLP', color='blue')
    axes[0].hist(mse_pos_gnn_safe, bins=bins_pos, alpha=0.5, label='GNN', color='red')
    axes[0].set_xscale('log')
    axes[0].set_xlabel('Position MSE per particle')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Position MSE Distribution')
    axes[0].legend()
    mse_vel_mlp_safe = np.clip(mse_vel_mlp, 1e-10, None)
    mse_vel_gnn_safe = np.clip(mse_vel_gnn, 1e-10, None)
    bins_vel = np.logspace(np.log10(min(np.min(mse_vel_mlp_safe), np.min(mse_vel_gnn_safe))), np.log10(max(np.max(mse_vel_mlp_safe), np.max(mse_vel_gnn_safe))), 50)
    axes[1].hist(mse_vel_mlp_safe, bins=bins_vel, alpha=0.5, label='MLP', color='blue')
    axes[1].hist(mse_vel_gnn_safe, bins=bins_vel, alpha=0.5, label='GNN', color='red')
    axes[1].set_xscale('log')
    axes[1].set_xlabel('Velocity MSE per particle')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Velocity MSE Distribution')
    axes[1].legend()
    err_e_mlp_safe = np.clip(err_e_mlp, 1e-10, None)
    err_e_gnn_safe = np.clip(err_e_gnn, 1e-10, None)
    axes[2].boxplot([err_e_mlp_safe, err_e_gnn_safe], labels=['MLP', 'GNN'])
    axes[2].set_yscale('log')
    axes[2].set_ylabel('Relative Energy Error')
    axes[2].set_title('Energy Conservation')
    plt.tight_layout()
    plot_filename = os.path.join(data_dir, 'comparative_analysis_5_' + str(int(time.time())) + '.png')
    plt.savefig(plot_filename, dpi=300)
    print('Comparative plots saved to ' + plot_filename)
if __name__ == '__main__':
    main()
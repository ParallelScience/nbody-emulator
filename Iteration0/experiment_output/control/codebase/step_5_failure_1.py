# filename: codebase/step_5.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import time
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from step_2 import InteractionNetwork

data_dir = 'data/'

class LeapfrogNode(nn.Module):
    def __init__(self, interaction_net, dt=0.01):
        super().__init__()
        self.interaction_net = interaction_net
        self.dt = dt

    def forward(self, pos, vel, steps, R, V):
        dt_eff = self.dt * (V / R).view(-1, 1, 1)
        for _ in range(steps):
            acc = self.interaction_net(pos)
            vel = vel + 0.5 * dt_eff * acc
            pos = pos + dt_eff * vel
            acc = self.interaction_net(pos)
            vel = vel + 0.5 * dt_eff * acc
        return pos, vel

class BaselineMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(300, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 300)
        )
    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        out = self.net(x)
        return out.view(batch_size, 50, 6)

def get_energy(pos, vel, eps=0.01):
    batch_size, N, _ = pos.shape
    kin = 0.5 * torch.sum(vel**2, dim=(1, 2))
    i, j = torch.triu_indices(N, N, offset=1)
    r_ij = pos[:, j, :] - pos[:, i, :]
    dist_sq = torch.sum(r_ij**2, dim=-1) + eps**2
    pot = -torch.sum(1.0 / torch.sqrt(dist_sq), dim=1)
    return kin + pot

def train_node(use_softening, model_path, train_traj, train_R, train_V, epochs=50):
    print('Training Neural ODE (softening=' + str(use_softening) + ')...')
    interaction_net = InteractionNetwork(hidden_dim=64, use_softening=use_softening)
    model = LeapfrogNode(interaction_net, dt=0.01)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    start_time = time.time()
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        for t_idx in range(1, 11):
            pos_in = train_traj[:, t_idx-1, :, :3]
            vel_in = train_traj[:, t_idx-1, :, 3:]
            pos_in = pos_in + torch.randn_like(pos_in) * 0.001
            vel_in = vel_in + torch.randn_like(vel_in) * 0.001
            pos_out, vel_out = model(pos_in, vel_in, steps=50, R=train_R, V=train_V)
            target_pos = train_traj[:, t_idx, :, :3]
            target_vel = train_traj[:, t_idx, :, 3:]
            step_loss = torch.nn.functional.mse_loss(pos_out, target_pos) + torch.nn.functional.mse_loss(vel_out, target_vel)
            step_loss.backward()
        optimizer.step()
    torch.save(model.state_dict(), model_path)
    print('Saved model to ' + model_path + '. Training took ' + str(round(time.time() - start_time, 2)) + 's')
    return model

def train_mlp(model_path, train_traj, epochs=200):
    print('Training Baseline MLP...')
    model = BaselineMLP()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    start_time = time.time()
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        x = train_traj[:, 0, :, :]
        y_true = train_traj[:, -1, :, :]
        y_pred = model(x)
        loss = torch.nn.functional.mse_loss(y_pred, y_true)
        loss.backward()
        optimizer.step()
    torch.save(model.state_dict(), model_path)
    print('Saved model to ' + model_path + '. Training took ' + str(round(time.time() - start_time, 2)) + 's')
    return model

def evaluate_models():
    trajectories = np.load(os.path.join(data_dir, 'trajectories.npy'))
    norm_coeffs = np.load(os.path.join(data_dir, 'normalization_coeffs.npy'))
    R = norm_coeffs[:, 0]
    V = norm_coeffs[:, 1]
    trajectories_norm = np.zeros_like(trajectories)
    for i in range(100):
        trajectories_norm[i, :, :, :3] = trajectories[i, :, :, :3] / R[i]
        trajectories_norm[i, :, :, 3:] = trajectories[i, :, :, 3:] / V[i]
    train_traj = torch.tensor(trajectories_norm[:80], dtype=torch.float32)
    test_traj = torch.tensor(trajectories_norm[80:], dtype=torch.float32)
    train_R = torch.tensor(R[:80], dtype=torch.float32)
    train_V = torch.tensor(V[:80], dtype=torch.float32)
    test_R = torch.tensor(R[80:], dtype=torch.float32)
    test_V = torch.tensor(V[80:], dtype=torch.float32)
    node_model_path = os.path.join(data_dir, 'node_model.pth')
    node_ablation_path = os.path.join(data_dir, 'node_ablation_model.pth')
    mlp_model_path = os.path.join(data_dir, 'mlp_model_retrained.pth')
    node_model = train_node(use_softening=True, model_path=node_model_path, train_traj=train_traj, train_R=train_R, train_V=train_V, epochs=50)
    node_ablation = train_node(use_softening=False, model_path=node_ablation_path, train_traj=train_traj, train_R=train_R, train_V=train_V, epochs=50)
    mlp_model = train_mlp(model_path=mlp_model_path, train_traj=train_traj, epochs=200)
    print('\nEvaluating models on test set...')
    results = []
    test_traj_phys = torch.tensor(trajectories[80:], dtype=torch.float32)
    pos_true = test_traj_phys[:, -1, :, :3]
    vel_true = test_traj_phys[:, -1, :, 3:]
    e_initial = get_energy(test_traj_phys[:, 0, :, :3], test_traj_phys[:, 0, :, 3:])
    node_model.eval()
    with torch.no_grad():
        pos_pred_norm, vel_pred_norm = node_model(test_traj[:, 0, :, :3], test_traj[:, 0, :, 3:], steps=500, R=test_R, V=test_V)
        pos_pred = pos_pred_norm * test_R.view(-1, 1, 1)
        vel_pred = vel_pred_norm * test_V.view(-1, 1, 1)
        mse_pos = torch.nn.functional.mse_loss(pos_pred, pos_true).item()
        mse_vel = torch.nn.functional.mse_loss(vel_pred, vel_true).item()
        e_pred_final = get_energy(pos_pred, vel_pred, eps=0.01)
        energy_error = torch.mean(torch.abs((e_pred_final - e_initial) / e_initial)).item()
        results.append({'Model': 'Neural ODE (Softening)', 'MSE Positions': mse_pos, 'MSE Velocities': mse_vel, 'Energy Error': energy_error})
    node_ablation.eval()
    with torch.no_grad():
        pos_pred_norm, vel_pred_norm = node_ablation(test_traj[:, 0, :, :3], test_traj[:, 0, :, 3:], steps=500, R=test_R, V=test_V)
        pos_pred = pos_pred_norm * test_R.view(-1, 1, 1)
        vel_pred = vel_pred_norm * test_V.view(-1, 1, 1)
        mse_pos = torch.nn.functional.mse_loss(pos_pred, pos_true).item()
        mse_vel = torch.nn.functional.mse_loss(vel_pred, vel_true).item()
        e_pred_final = get_energy(pos_pred, vel_pred, eps=0.01)
        energy_error = torch.mean(torch.abs((e_pred_final - e_initial) / e_initial)).item()
        results.append({'Model': 'Neural ODE (No Softening)', 'MSE Positions': mse_pos, 'MSE Velocities': mse_vel, 'Energy Error': energy_error})
    mlp_model.eval()
    with torch.no_grad():
        pred_norm = mlp_model(test_traj[:, 0, :, :])
        pos_pred_norm = pred_norm[:, :, :3]
        vel_pred_norm = pred_norm[:, :, 3:]
        pos_pred = pos_pred_norm * test_R.view(-1, 1, 1)
        vel_pred = vel_pred_norm * test_V.view(-1, 1, 1)
        mse_pos = torch.nn.functional.mse_loss(pos_pred, pos_true).item()
        mse_vel = torch.nn.functional.mse_loss(vel_pred, vel_true).item()
        e_pred_final = get_energy(pos_pred, vel_pred, eps=0.01)
        energy_error = torch.mean(torch.abs((e_pred_final - e_initial) / e_initial)).item()
        results.append({'Model': 'MLP Baseline', 'MSE Positions': mse_pos, 'MSE Velocities': mse_vel, 'Energy Error': energy_error})
    df = pd.DataFrame(results)
    print('\n--- Comparison Statistics ---')
    print(df.to_string(index=False))
    csv_path = os.path.join(data_dir, 'model_comparison.csv')
    df.to_csv(csv_path, index=False)
    print('\nComparison statistics saved to ' + csv_path)

if __name__ == '__main__':
    evaluate_models()
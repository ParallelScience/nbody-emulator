# filename: codebase/step_5.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
sys.path.insert(0, "/home/node/data/compsep_data/")
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from step_1 import generate_plummer_sphere
from step_2 import InteractionNetwork
class MLPForceField(nn.Module):
    def __init__(self, n_particles=50, hidden_dim=256):
        super(MLPForceField, self).__init__()
        self.net = nn.Sequential(nn.Linear(n_particles * 6, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, n_particles * 3))
    def forward(self, pos, vel):
        batch_size = pos.shape[0]
        x = torch.cat([pos.view(batch_size, -1), vel.view(batch_size, -1)], dim=-1)
        a = self.net(x)
        return a.view(batch_size, -1, 3)
def train_mlp():
    print("Training baseline MLP force field...")
    data_path = 'data/intermediate_snapshots.npz'
    splits_path = 'data/data_splits.npz'
    data = np.load(data_path)
    pos = data['pos'].transpose(1, 0, 2, 3)
    vel = data['vel'].transpose(1, 0, 2, 3)
    acc = data['acc'].transpose(1, 0, 2, 3)
    splits = np.load(splits_path)
    train_idx = splits['train_idx']
    train_pos = pos[train_idx].reshape(-1, 50, 3)
    train_vel = vel[train_idx].reshape(-1, 50, 3)
    train_acc = acc[train_idx].reshape(-1, 50, 3)
    device = torch.device('cpu')
    train_dataset = TensorDataset(torch.tensor(train_pos, dtype=torch.float32), torch.tensor(train_vel, dtype=torch.float32), torch.tensor(train_acc, dtype=torch.float32))
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    model = MLPForceField(n_particles=50, hidden_dim=256).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    epochs = 50
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for p, v, a in train_loader:
            p, v, a = p.to(device), v.to(device), a.to(device)
            optimizer.zero_grad()
            pred_a = model(p, v)
            loss = criterion(pred_a, a)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * p.size(0)
        train_loss /= len(train_loader.dataset)
    torch.save(model.state_dict(), 'data/mlp_force_field.pth')
    print("MLP force field trained and saved to data/mlp_force_field.pth")
    return model
def compute_energy(pos, vel, eps=0.01):
    ke = 0.5 * np.sum(vel**2)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=-1) + eps**2
    idx = np.triu_indices(pos.shape[0], k=1)
    pe = -np.sum(1.0 / np.sqrt(dist_sq[idx]))
    return ke + pe
def leapfrog_step(pos, vel, model, dt):
    with torch.no_grad():
        acc = model(pos, vel)
        v_half = vel + 0.5 * dt * acc
        pos_new = pos + dt * v_half
        acc_new = model(pos_new, v_half)
        vel_new = v_half + 0.5 * dt * acc_new
    return pos_new, vel_new
def rk4_step(pos, vel, model, dt):
    with torch.no_grad():
        v_k1 = vel
        a_k1 = model(pos, v_k1)
        p2 = pos + 0.5 * dt * v_k1
        v2 = vel + 0.5 * dt * a_k1
        a_k2 = model(p2, v2)
        p3 = pos + 0.5 * dt * v2
        v3 = vel + 0.5 * dt * a_k2
        a_k3 = model(p3, v3)
        p4 = pos + dt * v3
        v4 = vel + dt * a_k3
        a_k4 = model(p4, v4)
        pos_new = pos + (dt / 6.0) * (v_k1 + 2*v2 + 2*v3 + v4)
        vel_new = vel + (dt / 6.0) * (a_k1 + 2*a_k2 + 2*a_k3 + a_k4)
    return pos_new, vel_new
def rollout(model, pos_0, vel_0, integrator, dt, n_steps):
    pos = torch.tensor(pos_0, dtype=torch.float32).unsqueeze(0)
    vel = torch.tensor(vel_0, dtype=torch.float32).unsqueeze(0)
    energies = []
    energies.append(compute_energy(pos_0, vel_0))
    save_interval = max(1, int(0.01 / dt))
    for step in range(1, n_steps + 1):
        if integrator == 'leapfrog':
            pos, vel = leapfrog_step(pos, vel, model, dt)
        elif integrator == 'rk4':
            pos, vel = rk4_step(pos, vel, model, dt)
        if step % save_interval == 0:
            e = compute_energy(pos[0].cpu().numpy(), vel[0].cpu().numpy())
            energies.append(e)
    return np.array(energies)
def calc_drift(energies):
    e0 = energies[0]
    valid_energies = energies[np.isfinite(energies)]
    if len(valid_energies) == 0:
        return np.nan
    return np.var(valid_energies / e0)
def main():
    if not os.path.exists('data/mlp_force_field.pth'):
        mlp_model = train_mlp()
    else:
        print("Loading pre-trained MLP force field...")
        mlp_model = MLPForceField(n_particles=50, hidden_dim=256)
        mlp_model.load_state_dict(torch.load('data/mlp_force_field.pth'))
    mlp_model.eval()
    print("Loading pre-trained Physics-Informed GNN...")
    gnn_model = InteractionNetwork(include_physics_prior=True, hidden_dim=64)
    gnn_model.load_state_dict(torch.load('data/gnn_physics_informed.pth'))
    gnn_model.eval()
    splits = np.load('data/data_splits.npz')
    test_idx = splits['test_idx'][0]
    pos_ic, vel_ic = generate_plummer_sphere(50, 100)
    pos_0 = pos_ic[test_idx]
    vel_0 = vel_ic[test_idx]
    print("\nRunning rollouts on test simulation ID: " + str(test_idx))
    print("Integrating GNN + Leapfrog (dt=0.01, 500 steps)...")
    energies_gnn_lf = rollout(gnn_model, pos_0, vel_0, 'leapfrog', dt=0.01, n_steps=500)
    print("Integrating GNN + RK4 (dt=0.001, 5000 steps)...")
    energies_gnn_rk4 = rollout(gnn_model, pos_0, vel_0, 'rk4', dt=0.001, n_steps=5000)
    print("Integrating MLP + Leapfrog (dt=0.01, 500 steps)...")
    energies_mlp_lf = rollout(mlp_model, pos_0, vel_0, 'leapfrog', dt=0.01, n_steps=500)
    print("Integrating MLP + RK4 (dt=0.001, 5000 steps)...")
    energies_mlp_rk4 = rollout(mlp_model, pos_0, vel_0, 'rk4', dt=0.001, n_steps=5000)
    drift_gnn_lf = calc_drift(energies_gnn_lf)
    drift_gnn_rk4 = calc_drift(energies_gnn_rk4)
    drift_mlp_lf = calc_drift(energies_mlp_lf)
    drift_mlp_rk4 = calc_drift(energies_mlp_rk4)
    print("\n" + "="*40)
    print("ENERGY DRIFT RESULTS")
    print("="*40)
    print("Metric: Variance of (E / E_0) over the trajectory")
    print("GNN + Leapfrog Drift: " + str(drift_gnn_lf))
    print("GNN + RK4 Drift:      " + str(drift_gnn_rk4))
    print("MLP + Leapfrog Drift: " + str(drift_mlp_lf))
    print("MLP + RK4 Drift:      " + str(drift_mlp_rk4))
    print("="*40 + "\n")
    time_axis = np.linspace(0, 5.0, len(energies_gnn_lf))
    plt.rcParams['text.usetex'] = False
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    ax1.plot(time_axis, energies_gnn_lf, label='GNN + Leapfrog', color='blue', linestyle='-')
    ax1.plot(time_axis, energies_gnn_rk4, label='GNN + RK4', color='cyan', linestyle='--')
    ax1.plot(time_axis, energies_mlp_lf, label='MLP + Leapfrog', color='red', linestyle='-')
    ax1.plot(time_axis, energies_mlp_rk4, label='MLP + RK4', color='orange', linestyle='--')
    gnn_min = min(np.nanmin(energies_gnn_lf), np.nanmin(energies_gnn_rk4))
    gnn_max = max(np.nanmax(energies_gnn_lf), np.nanmax(energies_gnn_rk4))
    margin = max(0.05 * abs(gnn_max), (gnn_max - gnn_min) * 2)
    ax1.set_ylim(gnn_min - margin, gnn_max + margin)
    ax1.set_xlabel('Time (T)')
    ax1.set_ylabel('Total Energy (E) [M L^2/T^2]')
    ax1.set_title('Total Energy vs Time')
    ax1.legend()
    ax1.grid(True, alpha=0.5)
    drifts = [drift_gnn_lf, drift_gnn_rk4, drift_mlp_lf, drift_mlp_rk4]
    labels = ['GNN+LF', 'GNN+RK4', 'MLP+LF', 'MLP+RK4']
    colors = ['blue', 'cyan', 'red', 'orange']
    ax2.bar(labels, drifts, color=colors, edgecolor='black')
    ax2.set_yscale('log')
    ax2.set_ylabel('Energy Drift (Var(E/E_0))')
    ax2.set_title('Energy Drift Comparison')
    for i, v in enumerate(drifts):
        ax2.text(i, v * 1.2, str(round(v, 4)), ha='center', va='bottom')
    plt.tight_layout()
    timestamp = str(int(time.time()))
    plot_filename = 'data/energy_drift_analysis_1_' + timestamp + '.png'
    plt.savefig(plot_filename, dpi=300)
    print("Plot saved to " + plot_filename)
if __name__ == '__main__':
    main()
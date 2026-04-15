# filename: codebase/step_6.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['text.usetex'] = False

def compute_accelerations(pos, eps=0.01):
    dx = pos.unsqueeze(1) - pos.unsqueeze(2)
    r2 = torch.sum(dx**2, dim=-1)
    r3 = (r2 + eps**2)**1.5
    a = dx / r3.unsqueeze(-1)
    return torch.sum(a, dim=2)

def leapfrog_integrate(pos, vel, dt=0.01, steps=500, save_interval=50, eps=0.01):
    trajectories_pos = [pos.clone()]
    trajectories_vel = [vel.clone()]
    a = compute_accelerations(pos, eps)
    for step in range(1, steps + 1):
        vel_half = vel + 0.5 * dt * a
        pos = pos + dt * vel_half
        a = compute_accelerations(pos, eps)
        vel = vel_half + 0.5 * dt * a
        if step % save_interval == 0:
            trajectories_pos.append(pos.clone())
            trajectories_vel.append(vel.clone())
    return torch.stack(trajectories_pos, dim=1), torch.stack(trajectories_vel, dim=1)

def compute_energy(pos, vel, eps=0.01):
    ke = 0.5 * torch.sum(vel**2, dim=(1, 2))
    dx = pos.unsqueeze(1) - pos.unsqueeze(2)
    r2 = torch.sum(dx**2, dim=-1)
    N = pos.shape[1]
    mask = torch.triu(torch.ones(N, N, device=pos.device), diagonal=1)
    r2_safe = r2 + torch.eye(N, device=pos.device).unsqueeze(0)
    pe = - torch.sum(mask.unsqueeze(0) / torch.sqrt(r2_safe + eps**2), dim=(1, 2))
    return ke + pe

def generate_plummer(N_sim=100, N_part=50):
    pos = torch.zeros(N_sim, N_part, 3)
    vel = torch.zeros(N_sim, N_part, 3)
    for i in range(N_sim):
        r = 1.0 / torch.sqrt(torch.rand(N_part)**(-2.0/3.0) - 1.0)
        theta = torch.acos(2.0 * torch.rand(N_part) - 1.0)
        phi = 2.0 * np.pi * torch.rand(N_part)
        pos[i, :, 0] = r * torch.sin(theta) * torch.cos(phi)
        pos[i, :, 1] = r * torch.sin(theta) * torch.sin(phi)
        pos[i, :, 2] = r * torch.cos(theta)
        v_esc = np.sqrt(2.0) * (1.0 + r**2)**(-0.25)
        v_mag = v_esc * torch.rand(N_part)
        v_theta = torch.acos(2.0 * torch.rand(N_part) - 1.0)
        v_phi = 2.0 * np.pi * torch.rand(N_part)
        vel[i, :, 0] = v_mag * torch.sin(v_theta) * torch.cos(v_phi)
        vel[i, :, 1] = v_mag * torch.sin(v_theta) * torch.sin(v_phi)
        vel[i, :, 2] = v_mag * torch.cos(v_theta)
        pos[i] -= torch.mean(pos[i], dim=0)
        vel[i] -= torch.mean(vel[i], dim=0)
        pe = compute_energy(pos[i:i+1], torch.zeros(1, N_part, 3), eps=0.01)[0]
        ke = 0.5 * torch.sum(vel[i]**2)
        q = torch.sqrt(0.5 * torch.abs(pe) / ke)
        vel[i] *= q
    return pos, vel

class GNNInteractionNetwork(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.epsilon = nn.Parameter(torch.tensor(0.05, dtype=torch.float32))
        self.edge_mlp = nn.Sequential(nn.Linear(1, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
        nn.init.zeros_(self.edge_mlp[-1].weight)
        nn.init.zeros_(self.edge_mlp[-1].bias)
    def forward(self, pos):
        batch_size, N, _ = pos.shape
        dx = pos.unsqueeze(1) - pos.unsqueeze(2)
        r2 = torch.sum(dx**2, dim=-1)
        mask = ~torch.eye(N, dtype=torch.bool, device=pos.device).unsqueeze(0)
        r2_safe = r2 + (~mask).float()
        phys_force_mag = 1.0 / (r2_safe + self.epsilon**2)**1.5
        edge_inputs = r2_safe.unsqueeze(-1)
        m_ij = self.edge_mlp(edge_inputs).squeeze(-1)
        force_mag = m_ij * phys_force_mag
        force_vectors = force_mag.unsqueeze(-1) * dx
        force_vectors = force_vectors * mask.unsqueeze(-1).float()
        return torch.sum(force_vectors, dim=2)

class ODESolver(nn.Module):
    def __init__(self, model, dt=0.01):
        super().__init__()
        self.model = model
        self.dt = dt

class SymplecticVerlet(ODESolver):
    def integrate_trajectory(self, pos, vel, t_eval):
        pos_traj = []
        vel_traj = []
        current_t = 0.0
        a = self.model(pos)
        for t_target in t_eval:
            steps = int(round((t_target.item() - current_t) / self.dt))
            for _ in range(steps):
                vel_half = vel + 0.5 * self.dt * a
                pos = pos + self.dt * vel_half
                a = self.model(pos)
                vel = vel_half + 0.5 * self.dt * a
            pos_traj.append(pos.clone())
            vel_traj.append(vel.clone())
            current_t = t_target.item()
        return torch.stack(pos_traj, dim=0), torch.stack(vel_traj, dim=0)

class RK4(ODESolver):
    def integrate_trajectory(self, pos, vel, t_eval):
        pos_traj = []
        vel_traj = []
        current_t = 0.0
        for t_target in t_eval:
            steps = int(round((t_target.item() - current_t) / self.dt))
            for _ in range(steps):
                k1_v = self.model(pos)
                k1_r = vel
                pos2 = pos + 0.5 * self.dt * k1_r
                vel2 = vel + 0.5 * self.dt * k1_v
                k2_v = self.model(pos2)
                k2_r = vel2
                pos3 = pos + 0.5 * self.dt * k2_r
                vel3 = vel + 0.5 * self.dt * k2_v
                k3_v = self.model(pos3)
                k3_r = vel3
                pos4 = pos + self.dt * k3_r
                vel4 = vel + self.dt * k3_v
                k4_v = self.model(pos4)
                k4_r = vel4
                pos = pos + (self.dt / 6.0) * (k1_r + 2*k2_r + 2*k3_r + k4_r)
                vel = vel + (self.dt / 6.0) * (k1_v + 2*k2_v + 2*k3_v + k4_v)
            pos_traj.append(pos.clone())
            vel_traj.append(vel.clone())
            current_t = t_target.item()
        return torch.stack(pos_traj, dim=0), torch.stack(vel_traj, dim=0)

class BaselineMLP(nn.Module):
    def __init__(self, n_particles=50, hidden_dim=512):
        super().__init__()
        self.n_particles = n_particles
        input_dim = n_particles * 6
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, input_dim))
    def forward(self, pos, vel):
        batch_size = pos.shape[0]
        x = torch.cat([pos.reshape(batch_size, -1), vel.reshape(batch_size, -1)], dim=1)
        out = self.net(x)
        pos_out = out[:, :self.n_particles*3].reshape(batch_size, self.n_particles, 3)
        vel_out = out[:, self.n_particles*3:].reshape(batch_size, self.n_particles, 3)
        return pos_out, vel_out

if __name__ == '__main__':
    ic_final_path = 'data/ic_final.npy'
    if os.path.exists(ic_final_path):
        ic_final = np.load(ic_final_path)
        if ic_final.dtype.names is not None:
            time_field = ic_final['time']
            ic_mask = time_field[:, 0] == 0
            pos_ic = torch.tensor(np.stack([ic_final['x'][ic_mask], ic_final['y'][ic_mask], ic_final['z'][ic_mask]], axis=-1), dtype=torch.float32)
            vel_ic = torch.tensor(np.stack([ic_final['vx'][ic_mask], ic_final['vy'][ic_mask], ic_final['vz'][ic_mask]], axis=-1), dtype=torch.float32)
        else:
            ic_mask = ic_final[:, 0, 12] == 0
            pos_ic = torch.tensor(ic_final[ic_mask, :, 0:3], dtype=torch.float32)
            vel_ic = torch.tensor(ic_final[ic_mask, :, 3:6], dtype=torch.float32)
    else:
        pos_ic, vel_ic = generate_plummer(100, 50)
    pos_traj, vel_traj = leapfrog_integrate(pos_ic, vel_ic, dt=0.01, steps=500, save_interval=50, eps=0.01)
    num_sims = pos_traj.shape[0]
    num_train = int(0.8 * num_sims)
    pos_train, vel_train = pos_traj[:num_train], vel_traj[:num_train]
    pos_test, vel_test = pos_traj[num_train:], vel_traj[num_train:]
    t_eval_train = torch.linspace(0.5, 2.5, 5)
    t_eval_rollout = torch.linspace(0.0, 5.0, 11)
    def train_model(model_type, epochs=40, lr=1e-3, lambda_energy=1e-3):
        gnn = GNNInteractionNetwork(hidden_dim=64)
        solver = SymplecticVerlet(gnn, dt=0.01) if model_type == 'symplectic' else RK4(gnn, dt=0.01)
        optimizer = optim.Adam(gnn.parameters(), lr=lr)
        dataset = torch.utils.data.TensorDataset(pos_train, vel_train)
        loader = torch.utils.data.DataLoader(dataset, batch_size=20, shuffle=True)
        history = {'loss_traj': [], 'eps': []}
        for epoch in range(epochs):
            total_loss_traj = 0.0
            for p, v in loader:
                p_init, v_init = p[:, 0], v[:, 0]
                p_target, v_target = p[:, 1:6], v[:, 1:6]
                optimizer.zero_grad()
                p_pred, v_pred = solver.integrate_trajectory(p_init, v_init, t_eval_train)
                p_pred, v_pred = p_pred.permute(1, 0, 2, 3), v_pred.permute(1, 0, 2, 3)
                loss_traj = nn.functional.mse_loss(p_pred, p_target) + nn.functional.mse_loss(v_pred, v_target)
                e_init = compute_energy(p_init, v_init, eps=gnn.epsilon)
                loss_energy = 0.0
                for i in range(p_pred.shape[1]):
                    e_pred = compute_energy(p_pred[:, i], v_pred[:, i], eps=gnn.epsilon)
                    loss_energy += nn.functional.mse_loss(e_pred, e_init)
                loss = loss_traj + lambda_energy * (loss_energy / p_pred.shape[1])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(gnn.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss_traj += loss_traj.item()
            history['loss_traj'].append(total_loss_traj / len(loader))
            history['eps'].append(gnn.epsilon.item())
        return gnn, solver, history
    gnn_symp, solver_symp, hist_symp = train_model('symplectic')
    gnn_rk4, solver_rk4, hist_rk4 = train_model('rk4')
    mlp = BaselineMLP(n_particles=50)
    optimizer_mlp = optim.Adam(mlp.parameters(), lr=1e-3)
    for epoch in range(100):
        for p, v in torch.utils.data.DataLoader(torch.utils.data.TensorDataset(pos_train, vel_train), batch_size=20, shuffle=True):
            optimizer_mlp.zero_grad()
            p_pred, v_pred = mlp(p[:, 0], v[:, 0])
            loss = nn.functional.mse_loss(p_pred, p[:, -1]) + nn.functional.mse_loss(v_pred, v[:, -1])
            loss.backward()
            optimizer_mlp.step()
    results = {}
    for N in [25, 40, 50]:
        p_test_N, v_test_N = pos_test[:, :, :N, :], vel_test[:, :, :N, :]
        p_init, v_init = p_test_N[:, 0], v_test_N[:, 0]
        energies_gt = torch.stack([compute_energy(p_test_N[:, i], v_test_N[:, i], eps=0.01) for i in range(11)], dim=1)
        with torch.no_grad():
            p_pred_symp, v_pred_symp = solver_symp.integrate_trajectory(p_init, v_init, t_eval_rollout[1:])
            p_pred_symp, v_pred_symp = torch.cat([p_init.unsqueeze(0), p_pred_symp], dim=0).permute(1, 0, 2, 3), torch.cat([v_init.unsqueeze(0), v_pred_symp], dim=0).permute(1, 0, 2, 3)
            energies_symp = torch.stack([compute_energy(p_pred_symp[:, i], v_pred_symp[:, i], eps=gnn_symp.epsilon.item()) for i in range(11)], dim=1)
            p_pred_rk4, v_pred_rk4 = solver_rk4.integrate_trajectory(p_init, v_init, t_eval_rollout[1:])
            p_pred_rk4, v_pred_rk4 = torch.cat([p_init.unsqueeze(0), p_pred_rk4], dim=0).permute(1, 0, 2, 3), torch.cat([v_init.unsqueeze(0), v_pred_rk4], dim=0).permute(1, 0, 2, 3)
            energies_rk4 = torch.stack([compute_energy(p_pred_rk4[:, i], v_pred_rk4[:, i], eps=gnn_rk4.epsilon.item()) for i in range(11)], dim=1)
            p_pred_mlp, v_pred_mlp = mlp(p_init, v_init) if N == 50 else (None, None)
        results[N] = {'gt': {'traj_p': p_test_N, 'energies': energies_gt}, 'symp': {'traj_p': p_pred_symp, 'energies': energies_symp, 'mse_pos': nn.functional.mse_loss(p_pred_symp[:, -1], p_test_N[:, -1]).item()}, 'rk4': {'traj_p': p_pred_rk4, 'energies': energies_rk4, 'mse_pos': nn.functional.mse_loss(p_pred_rk4[:, -1], p_test_N[:, -1]).item()}, 'mlp': {'mse_pos': nn.functional.mse_loss(p_pred_mlp, p_test_N[:, -1]).item() if N == 50 else None}}
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    ax = axs[0, 0]
    for i, p_idx in enumerate([0, 1, 2]):
        ax.plot(results[50]['gt']['traj_p'][0, :, p_idx, 0], results[50]['gt']['traj_p'][0, :, p_idx, 1], 'k-', label='GT' if i==0 else "", linewidth=2)
        ax.plot(results[50]['symp']['traj_p'][0, :, p_idx, 0], results[50]['symp']['traj_p'][0, :, p_idx, 1], 'b--', label='Symp' if i==0 else "", linewidth=2)
        ax.plot(results[50]['rk4']['traj_p'][0, :, p_idx, 0], results[50]['rk4']['traj_p'][0, :, p_idx, 1], 'r:', label='RK4' if i==0 else "", linewidth=2)
    ax.set_title('Trajectory Comparison (N=50)'); ax.legend(); ax.grid(True)
    ax = axs[0, 1]
    for model, color, label in [('gt', 'k-', 'GT'), ('symp', 'b--', 'Symp'), ('rk4', 'r:', 'RK4')]:
        e = results[50][model]['energies'].numpy(); ax.plot(np.linspace(0, 5, 11), np.mean(np.abs(e - e[:, 0:1]) / np.abs(e[:, 0:1]), axis=0), color, label=label)
    ax.set_yscale('log'); ax.set_title('Energy Drift'); ax.legend(); ax.grid(True)
    ax = axs[1, 0]
    ax.plot(hist_symp['eps'], 'b-', label='Symp'); ax.plot(hist_rk4['eps'], 'r-', label='RK4'); ax.axhline(0.01, color='k', linestyle='--'); ax.set_title('Epsilon Convergence'); ax.legend(); ax.grid(True)
    ax = axs[1, 1]
    N_vals = [25, 40, 50]
    ax.plot(N_vals, [results[N]['symp']['mse_pos'] for N in N_vals], 'b-o', label='Symp'); ax.plot(N_vals, [results[N]['rk4']['mse_pos'] for N in N_vals], 'r-s', label='RK4'); ax.plot([50], [results[50]['mlp']['mse_pos']], 'g^', label='MLP'); ax.set_title('Generalization'); ax.legend(); ax.grid(True)
    plt.tight_layout(); plt.savefig('data/analysis_summary_fixed_' + str(int(time.time())) + '.png', dpi=300)
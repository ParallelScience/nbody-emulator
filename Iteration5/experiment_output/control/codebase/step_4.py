# filename: codebase/step_4.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import time
import json
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from step_1 import generate_plummer_sphere, compute_accelerations
from step_2 import HNNPotential
def compute_force_eval(pos, potential_model):
    with torch.enable_grad():
        pos = pos.detach().requires_grad_(True)
        U = potential_model(pos)
        force = -torch.autograd.grad(outputs=U, inputs=pos, grad_outputs=torch.ones_like(U), create_graph=False)[0]
    return force.detach()
class EvalNeuralLeapfrog(nn.Module):
    def __init__(self, potential_model, dt=0.01, steps=500):
        super().__init__()
        self.potential_model = potential_model
        self.dt = dt
        self.steps = steps
    def forward(self, pos, vel):
        pos = pos.clone()
        vel = vel.clone()
        force = compute_force_eval(pos, self.potential_model)
        for _ in range(self.steps):
            vel_half = vel + 0.5 * self.dt * force
            pos = pos + self.dt * vel_half
            force = compute_force_eval(pos, self.potential_model)
            vel = vel_half + 0.5 * self.dt * force
        return pos, vel
def run_true_leapfrog(pos, vel, mass, dt=0.01, steps=500, eps=0.01):
    pos = pos.copy()
    vel = vel.copy()
    acc = compute_accelerations(pos, mass, eps)
    for _ in range(steps):
        vel_half = vel + 0.5 * dt * acc
        pos = pos + dt * vel_half
        acc = compute_accelerations(pos, mass, eps)
        vel = vel_half + 0.5 * dt * acc
    return pos, vel
def run_point_mass_leapfrog(pos, vel, mass, dt=0.01, steps=500, eps=0.01):
    pos = pos.copy()
    vel = vel.copy()
    M_total = np.sum(mass, axis=1, keepdims=True)
    def get_acc(p):
        dist_sq = np.sum(p**2, axis=-1, keepdims=True) + eps**2
        dist_cube = dist_sq ** 1.5
        return -p * M_total[..., np.newaxis] / dist_cube
    acc = get_acc(pos)
    for _ in range(steps):
        vel_half = vel + 0.5 * dt * acc
        pos = pos + dt * vel_half
        acc = get_acc(pos)
        vel = vel_half + 0.5 * dt * acc
    return pos, vel
def true_hamiltonian(pos, vel, mass, eps=0.01):
    B, N, _ = pos.shape
    ke = 0.5 * np.sum(mass[:, :, np.newaxis] * vel**2, axis=(1, 2))
    pe = np.zeros(B)
    for i in range(N):
        for j in range(i+1, N):
            dx = pos[:, i, :] - pos[:, j, :]
            dist = np.sqrt(np.sum(dx**2, axis=-1) + eps**2)
            pe -= (mass[:, i] * mass[:, j]) / dist
    return ke + pe
def compute_jacobian_determinant(potential_model, device):
    pos = torch.randn(1, 2, 3, device=device, dtype=torch.float32)
    vel = torch.randn(1, 2, 3, device=device, dtype=torch.float32)
    pos.requires_grad_(True)
    vel.requires_grad_(True)
    dt = 0.01
    def step_fn(p, v):
        U = potential_model(p)
        f1 = -torch.autograd.grad(U, p, create_graph=True)[0]
        v_half = v + 0.5 * dt * f1
        p_next = p + dt * v_half
        U_next = potential_model(p_next)
        f2 = -torch.autograd.grad(U_next, p_next, create_graph=True)[0]
        v_next = v_half + 0.5 * dt * f2
        return p_next, v_next
    p_next, v_next = step_fn(pos, vel)
    out = torch.cat([p_next.view(-1), v_next.view(-1)])
    J = torch.zeros(12, 12)
    for i in range(12):
        grad_out = torch.zeros(12, device=device)
        grad_out[i] = 1.0
        grads = torch.autograd.grad(out, [pos, vel], grad_outputs=grad_out, retain_graph=True)
        J[i, :6] = grads[0].view(-1).cpu()
        J[i, 6:] = grads[1].view(-1).cpu()
    det = torch.det(J).item()
    return det
if __name__ == '__main__':
    data_dir = 'data/'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    stats = np.load(os.path.join(data_dir, 'normalization_stats.npz'))
    mean = stats['mean']
    std = stats['std']
    val_data_norm = np.load(os.path.join(data_dir, 'val_trajectories.npy'))
    val_data = val_data_norm * (std + 1e-8) + mean
    pos_init_val = val_data[:, 0, :, :3]
    vel_init_val = val_data[:, 0, :, 3:]
    pos_final_val = val_data[:, -1, :, :3]
    vel_final_val = val_data[:, -1, :, 3:]
    mass_val = np.ones((pos_init_val.shape[0], pos_init_val.shape[1]))
    potential_model = HNNPotential(hidden_dim=64, eps=0.01).to(device)
    potential_model.load_state_dict(torch.load(os.path.join(data_dir, 'hnn_model.pth'), map_location=device))
    potential_model.eval()
    eval_integrator = EvalNeuralLeapfrog(potential_model, dt=0.01, steps=500)
    pos_init_t = torch.tensor(pos_init_val, dtype=torch.float32).to(device)
    vel_init_t = torch.tensor(vel_init_val, dtype=torch.float32).to(device)
    pos_pred_val_t, vel_pred_val_t = eval_integrator(pos_init_t, vel_init_t)
    pos_pred_val = pos_pred_val_t.cpu().numpy()
    vel_pred_val = vel_pred_val_t.cpu().numpy()
    mse_pos_50 = np.mean((pos_pred_val - pos_final_val)**2)
    mse_vel_50 = np.mean((vel_pred_val - vel_final_val)**2)
    pos_pm_val, vel_pm_val = run_point_mass_leapfrog(pos_init_val, vel_init_val, mass_val, dt=0.01, steps=500)
    mse_pos_pm_50 = np.mean((pos_pm_val - pos_final_val)**2)
    mse_vel_pm_50 = np.mean((vel_pm_val - vel_final_val)**2)
    def evaluate_transfer(N, n_sims=20):
        pos_init, vel_init, mass = generate_plummer_sphere(N, n_sims)
        pos_true, vel_true = run_true_leapfrog(pos_init, vel_init, mass, dt=0.01, steps=500)
        pos_init_t = torch.tensor(pos_init, dtype=torch.float32).to(device)
        vel_init_t = torch.tensor(vel_init, dtype=torch.float32).to(device)
        pos_pred_t, vel_pred_t = eval_integrator(pos_init_t, vel_init_t)
        pos_pred = pos_pred_t.cpu().numpy()
        vel_pred = vel_pred_t.cpu().numpy()
        pos_pm, vel_pm = run_point_mass_leapfrog(pos_init, vel_init, mass, dt=0.01, steps=500)
        mse_pos = np.mean((pos_pred - pos_true)**2)
        mse_vel = np.mean((vel_pred - vel_true)**2)
        mse_pos_pm = np.mean((pos_pm - pos_true)**2)
        mse_vel_pm = np.mean((vel_pm - vel_true)**2)
        return mse_pos, mse_vel, mse_pos_pm, mse_vel_pm
    mse_pos_25, mse_vel_25, mse_pos_pm_25, mse_vel_pm_25 = evaluate_transfer(25)
    mse_pos_100, mse_vel_100, mse_pos_pm_100, mse_vel_pm_100 = evaluate_transfer(100)
    n_traj = 5
    pos_t = torch.tensor(pos_init_val[:n_traj], dtype=torch.float32).to(device)
    vel_t = torch.tensor(vel_init_val[:n_traj], dtype=torch.float32).to(device)
    H_vals = []
    pos = pos_t.clone()
    vel = vel_t.clone()
    force = compute_force_eval(pos, potential_model)
    H_vals.append(true_hamiltonian(pos.cpu().numpy(), vel.cpu().numpy(), mass_val[:n_traj]))
    dt = 0.01
    for _ in range(500):
        vel_half = vel + 0.5 * dt * force
        pos = pos + dt * vel_half
        force = compute_force_eval(pos, potential_model)
        vel = vel_half + 0.5 * dt * force
        if (_ + 1) % 50 == 0:
            H_vals.append(true_hamiltonian(pos.cpu().numpy(), vel.cpu().numpy(), mass_val[:n_traj]))
    H_vals = np.array(H_vals)
    delta_H = np.abs(H_vals[-1] - H_vals[0])
    mean_delta_H = np.mean(delta_H)
    integrator_bwd = EvalNeuralLeapfrog(potential_model, dt=-0.01, steps=500)
    pos_bwd_t, vel_bwd_t = integrator_bwd(pos_pred_val_t, vel_pred_val_t)
    pos_bwd = pos_bwd_t.cpu().numpy()
    vel_bwd = vel_bwd_t.cpu().numpy()
    mse_rev_pos = np.mean((pos_bwd - pos_init_val)**2)
    mse_rev_vel = np.mean((vel_bwd - vel_init_val)**2)
    det_J = compute_jacobian_determinant(potential_model, device)
    results = {'mse_pos_50': float(mse_pos_50), 'mse_vel_50': float(mse_vel_50), 'mse_pos_pm_50': float(mse_pos_pm_50), 'mse_vel_pm_50': float(mse_vel_pm_50), 'mse_pos_25': float(mse_pos_25), 'mse_vel_25': float(mse_vel_25), 'mse_pos_pm_25': float(mse_pos_pm_25), 'mse_vel_pm_25': float(mse_vel_pm_25), 'mse_pos_100': float(mse_pos_100), 'mse_vel_100': float(mse_vel_100), 'mse_pos_pm_100': float(mse_pos_pm_100), 'mse_vel_pm_100': float(mse_vel_pm_100), 'mean_delta_H': float(mean_delta_H), 'mse_rev_pos': float(mse_rev_pos), 'mse_rev_vel': float(mse_rev_vel), 'jacobian_det': float(det_J)}
    results_path = os.path.join(data_dir, 'evaluation_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    plt.rcParams['text.usetex'] = False
    fig, ax = plt.subplots(figsize=(8, 6))
    labels = ['N=25', 'N=50', 'N=100']
    model_mse = [mse_pos_25, mse_pos_50, mse_pos_100]
    baseline_mse = [mse_pos_pm_25, mse_pos_pm_50, mse_pos_pm_100]
    x = np.arange(len(labels))
    width = 0.35
    ax.bar(x - width/2, model_mse, width, label='HNN Model')
    ax.bar(x + width/2, baseline_mse, width, label='Point-Mass Baseline')
    ax.set_ylabel('Position MSE (Log Scale)')
    ax.set_title('Trajectory Reconstruction Error by Particle Count')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_yscale('log')
    ax.legend()
    fig.tight_layout()
    timestamp = int(time.time())
    plot1_path = os.path.join(data_dir, 'mse_comparison_1_' + str(timestamp) + '.png')
    fig.savefig(plot1_path, dpi=300)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 6))
    time_steps = np.linspace(0, 5.0, 11)
    for i in range(n_traj):
        ax.plot(time_steps, H_vals[:, i] - H_vals[0, i], marker='o', label='Sim ' + str(i+1))
    ax.set_xlabel('Time (T)')
    ax.set_ylabel('H(t) - H(0) [Energy Units]')
    ax.set_title('Hamiltonian Deviation over Time (HNN Model)')
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    plot2_path = os.path.join(data_dir, 'hamiltonian_deviation_2_' + str(timestamp) + '.png')
    fig.savefig(plot2_path, dpi=300)
    plt.close(fig)
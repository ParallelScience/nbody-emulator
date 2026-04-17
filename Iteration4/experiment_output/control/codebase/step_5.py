# filename: codebase/step_5.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import torch
import numpy as np
from step_1 import generate_plummer_sphere, get_accelerations, compute_energies_vectorized
from step_2 import ResidualGNN
from step_3 import NBodyODE, odeint, compute_energies

def generate_test_set(N_sim, N_part, dt=0.01, n_steps=500, snapshot_interval=50, eps=0.01):
    np.random.seed(42 + N_part)
    pos_list = []
    vel_list = []
    for _ in range(N_sim):
        p, v = generate_plummer_sphere(N_part)
        pos_list.append(p)
        vel_list.append(v)
    pos = np.array(pos_list)
    vel = np.array(vel_list)
    initial_conditions = np.concatenate([pos, vel], axis=2)
    snapshots_pos, snapshots_vel, snapshots_acc, snapshots_t = [], [], [], []
    acc = get_accelerations(pos, eps)
    for step in range(1, n_steps + 1):
        v_half = vel + 0.5 * dt * acc
        pos = pos + dt * v_half
        acc = get_accelerations(pos, eps)
        vel = v_half + 0.5 * dt * acc
        if step % snapshot_interval == 0:
            snapshots_pos.append(pos.copy())
            snapshots_vel.append(vel.copy())
            snapshots_acc.append(acc.copy())
            snapshots_t.append(step * dt)
    snapshots_pos = np.array(snapshots_pos).transpose(1, 0, 2, 3)
    snapshots_vel = np.array(snapshots_vel).transpose(1, 0, 2, 3)
    snapshots_acc = np.array(snapshots_acc).transpose(1, 0, 2, 3)
    snapshots_t = np.array(snapshots_t)
    ke_initial, pe_initial = compute_energies_vectorized(initial_conditions[:, :, 0:3], initial_conditions[:, :, 3:6], eps)
    energy_initial = ke_initial + pe_initial
    return {'pos': snapshots_pos, 'vel': snapshots_vel, 'acc': snapshots_acc, 't': snapshots_t, 'pos_init': initial_conditions[:, :, 0:3], 'vel_init': initial_conditions[:, :, 3:6], 'energy_init': energy_initial}

def evaluate_generalization(model, test_data, device):
    pos_init = torch.tensor(test_data['pos_init'], dtype=torch.float32, device=device)
    vel_init = torch.tensor(test_data['vel_init'], dtype=torch.float32, device=device)
    pos_true = torch.tensor(test_data['pos'], dtype=torch.float32, device=device)
    vel_true = torch.tensor(test_data['vel'], dtype=torch.float32, device=device)
    t_eval = torch.tensor(test_data['t'], dtype=torch.float32, device=device)
    t_eval_with_zero = torch.cat([torch.tensor([0.0], device=device), t_eval])
    ode_func = NBodyODE(model)
    z0 = torch.cat([pos_init, vel_init], dim=-1)
    with torch.no_grad():
        z_pred = odeint(ode_func, z0, t_eval_with_zero, method='dopri5', rtol=1e-3, atol=1e-4)
    z_pred_snapshots = z_pred[1:]
    pos_pred = z_pred_snapshots[..., :3]
    vel_pred = z_pred_snapshots[..., 3:]
    pos_pred_T5 = pos_pred[-1]
    vel_pred_T5 = vel_pred[-1]
    pos_true_T5 = pos_true[-1]
    vel_true_T5 = vel_true[-1]
    pos_mse_T5 = torch.nn.functional.mse_loss(pos_pred_T5, pos_true_T5).item()
    vel_mse_T5 = torch.nn.functional.mse_loss(vel_pred_T5, vel_true_T5).item()
    energies_pred = []
    for i in range(len(t_eval)):
        e = compute_energies(pos_pred[i], vel_pred[i], eps=0.01)
        energies_pred.append(e)
    energies_pred = torch.stack(energies_pred)
    energy_init = compute_energies(pos_init, vel_init, eps=0.01)
    energy_drift = torch.abs((energies_pred - energy_init.unsqueeze(0)) / energy_init.unsqueeze(0))
    mean_energy_drift = energy_drift.mean().item()
    max_energy_drift = energy_drift.max().item()
    return {'pos_mse_T5': pos_mse_T5, 'vel_mse_T5': vel_mse_T5, 'mean_energy_drift': mean_energy_drift, 'max_energy_drift': max_energy_drift, 'pos_pred': pos_pred.cpu().numpy(), 'vel_pred': vel_pred.cpu().numpy(), 'energies_pred': energies_pred.cpu().numpy(), 'energy_init': energy_init.cpu().numpy()}

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_dir = 'data/'
    model = ResidualGNN(hidden_dim=32, eps=0.01).to(device)
    model_path = os.path.join(data_dir, 'trained_model.pth')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    N_sim = 10
    N_values = [25, 100]
    results = {}
    for N in N_values:
        test_data = generate_test_set(N_sim=N_sim, N_part=N)
        metrics = evaluate_generalization(model, test_data, device)
        results['N_' + str(N)] = metrics
    save_path = os.path.join(data_dir, 'generalization_metrics.npy')
    np.save(save_path, results)
    print('Saved generalization metrics to ' + save_path)
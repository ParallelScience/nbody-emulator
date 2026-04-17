# filename: codebase/step_4.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import torch
import numpy as np
from step_2 import ResidualGNN
from step_3 import NBodyODE, odeint, compute_energies
from step_1 import get_accelerations
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device: ' + str(device))
    data_dir = 'data/'
    test_data = np.load(os.path.join(data_dir, 'test_data.npz'))
    pos_init = torch.tensor(test_data['pos_init'], dtype=torch.float32, device=device)
    vel_init = torch.tensor(test_data['vel_init'], dtype=torch.float32, device=device)
    pos_true = torch.tensor(test_data['pos'], dtype=torch.float32, device=device)
    vel_true = torch.tensor(test_data['vel'], dtype=torch.float32, device=device)
    t_eval = torch.tensor(test_data['t'], dtype=torch.float32, device=device)
    t_eval_with_zero = torch.cat([torch.tensor([0.0], device=device), t_eval])
    model = ResidualGNN(hidden_dim=32, eps=0.01).to(device)
    model.load_state_dict(torch.load(os.path.join(data_dir, 'trained_model.pth'), map_location=device))
    model.eval()
    ode_func = NBodyODE(model)
    print('\n--- 1. Forward Integration to T=5.0 ---')
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
    print('Model vs Ground Truth at T=5.0:')
    print('  Position MSE: ' + str(pos_mse_T5))
    print('  Velocity MSE: ' + str(vel_mse_T5))
    print('\n--- 2. Time-Reversibility Analysis ---')
    z_T5 = z_pred[-1]
    pos_T5 = z_T5[..., :3]
    vel_T5 = z_T5[..., 3:]
    z0_back = torch.cat([pos_T5, -vel_T5], dim=-1)
    t_back = torch.linspace(0, 5.0, 2, device=device)
    with torch.no_grad():
        z_back_pred = odeint(ode_func, z0_back, t_back, method='dopri5', rtol=1e-3, atol=1e-4)
    z_recovered = z_back_pred[-1]
    pos_recovered = z_recovered[..., :3]
    vel_recovered = -z_recovered[..., 3:]
    reversibility_pos_mse = torch.nn.functional.mse_loss(pos_recovered, pos_init).item()
    reversibility_vel_mse = torch.nn.functional.mse_loss(vel_recovered, vel_init).item()
    print('Forward-Backward Integration Error (Recovered vs Initial):')
    print('  Position MSE: ' + str(reversibility_pos_mse))
    print('  Velocity MSE: ' + str(reversibility_vel_mse))
    print('\n--- 3. Extended Rollout to T=10.0 ---')
    dt = 0.01
    n_steps_10 = 1000
    eps = 0.01
    pos_lf = test_data['pos_init'].copy()
    vel_lf = test_data['vel_init'].copy()
    acc_lf = get_accelerations(pos_lf, eps)
    pos_lf_traj = []
    vel_lf_traj = []
    for step in range(1, n_steps_10 + 1):
        v_half = vel_lf + 0.5 * dt * acc_lf
        pos_lf = pos_lf + dt * v_half
        acc_lf = get_accelerations(pos_lf, eps)
        vel_lf = v_half + 0.5 * dt * acc_lf
        if step % 50 == 0:
            pos_lf_traj.append(pos_lf.copy())
            vel_lf_traj.append(vel_lf.copy())
    pos_lf_traj = np.array(pos_lf_traj)
    vel_lf_traj = np.array(vel_lf_traj)
    t_eval_10 = torch.linspace(0, 10.0, 21, device=device)
    with torch.no_grad():
        z_pred_10 = odeint(ode_func, z0, t_eval_10, method='dopri5', rtol=1e-3, atol=1e-4)
    z_pred_10_snapshots = z_pred_10[1:]
    pos_pred_10 = z_pred_10_snapshots[..., :3]
    vel_pred_10 = z_pred_10_snapshots[..., 3:]
    pos_lf_traj_t = torch.tensor(pos_lf_traj, dtype=torch.float32, device=device)
    vel_lf_traj_t = torch.tensor(vel_lf_traj, dtype=torch.float32, device=device)
    divergence_pos_mse = torch.nn.functional.mse_loss(pos_pred_10, pos_lf_traj_t).item()
    divergence_vel_mse = torch.nn.functional.mse_loss(vel_pred_10, vel_lf_traj_t).item()
    energies_10 = []
    for i in range(20):
        e = compute_energies(pos_pred_10[i], vel_pred_10[i], eps=0.01)
        energies_10.append(e)
    energies_10 = torch.stack(energies_10)
    energy_init = compute_energies(pos_init, vel_init, eps=0.01)
    energy_drift = torch.abs((energies_10 - energy_init.unsqueeze(0)) / energy_init.unsqueeze(0))
    mean_energy_drift = energy_drift.mean().item()
    max_energy_drift = energy_drift.max().item()
    print('Trajectory Divergence (Model vs Leapfrog up to T=10.0):')
    print('  Position MSE: ' + str(divergence_pos_mse))
    print('  Velocity MSE: ' + str(divergence_vel_mse))
    print('Energy Conservation (Model up to T=10.0):')
    print('  Mean Relative Energy Drift: ' + str(mean_energy_drift))
    print('  Max Relative Energy Drift: ' + str(max_energy_drift))
    reversibility_stability_metrics = {'reversibility_pos_mse': reversibility_pos_mse, 'reversibility_vel_mse': reversibility_vel_mse, 'divergence_pos_mse': divergence_pos_mse, 'divergence_vel_mse': divergence_vel_mse, 'mean_energy_drift': mean_energy_drift, 'max_energy_drift': max_energy_drift, 'energies_10': energies_10.cpu().numpy(), 'energy_init': energy_init.cpu().numpy(), 'pos_pred_10': pos_pred_10.cpu().numpy(), 'pos_lf_traj': pos_lf_traj}
    np.save(os.path.join(data_dir, 'reversibility_stability_metrics.npy'), reversibility_stability_metrics)
    print('Saved reversibility and stability metrics to data/reversibility_stability_metrics.npy')
    print('\n--- 4. Density-Dependent Error Analysis ---')
    B, N = pos_init.shape[0], pos_init.shape[1]
    k = 5
    densities = []
    pos_errors = []
    vel_errors = []
    for t_idx in range(10):
        p_true = pos_true[t_idx]
        p_pred = pos_pred[t_idx]
        v_true = vel_true[t_idx]
        v_pred = vel_pred[t_idx]
        dx = p_true.unsqueeze(2) - p_true.unsqueeze(1)
        dist = torch.sqrt(torch.sum(dx**2, dim=-1) + 1e-8)
        sorted_dist, _ = torch.sort(dist, dim=-1)
        r5 = sorted_dist[:, :, k]
        density = 1.0 / (r5**3 + 1e-8)
        p_err = torch.sum((p_pred - p_true)**2, dim=-1)
        v_err = torch.sum((v_pred - v_true)**2, dim=-1)
        densities.append(density.cpu().numpy())
        pos_errors.append(p_err.cpu().numpy())
        vel_errors.append(v_err.cpu().numpy())
    densities = np.array(densities)
    pos_errors = np.array(pos_errors)
    vel_errors = np.array(vel_errors)
    density_error_metrics = {'densities': densities, 'pos_errors': pos_errors, 'vel_errors': vel_errors}
    np.save(os.path.join(data_dir, 'density_error_metrics.npy'), density_error_metrics)
    print('Saved density and error metrics to data/density_error_metrics.npy')
    print('Density Statistics (over all snapshots):')
    print('  Mean Density: ' + str(densities.mean()))
    print('  Max Density: ' + str(densities.max()))
    print('  Min Density: ' + str(densities.min()))
    print('Error Statistics (over all snapshots):')
    print('  Mean Position Squared Error: ' + str(pos_errors.mean()))
    print('  Mean Velocity Squared Error: ' + str(vel_errors.mean()))
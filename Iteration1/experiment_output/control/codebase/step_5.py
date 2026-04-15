# filename: codebase/step_5.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import torch
import torch.nn as nn
from step_1 import NBodyDataset
from step_2 import GNNInteractionNetwork
from step_3 import SymplecticVerlet, RK4, compute_energy
from step_4 import BaselineMLP

def evaluate_rollout():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = torch.load('data/trajectories.pt', weights_only=True)
    pos_all = data['pos']
    vel_all = data['vel']
    t_eval = data['t'].to(device)
    pos_test = pos_all[80:].to(device)
    vel_test = vel_all[80:].to(device)
    gnn_symp = GNNInteractionNetwork(hidden_dim=64).to(device)
    gnn_symp.load_state_dict(torch.load('data/gnn_symplectic.pt', map_location=device, weights_only=True))
    gnn_symp.eval()
    gnn_rk4 = GNNInteractionNetwork(hidden_dim=64).to(device)
    gnn_rk4.load_state_dict(torch.load('data/gnn_rk4.pt', map_location=device, weights_only=True))
    gnn_rk4.eval()
    mlp = BaselineMLP(n_particles=50).to(device)
    mlp.load_state_dict(torch.load('data/model_mlp.pt', map_location=device, weights_only=True))
    mlp.eval()
    solver_symp = SymplecticVerlet(gnn_symp, dt=0.01)
    solver_rk4 = RK4(gnn_rk4, dt=0.01)
    results = {}
    for N in [25, 40, 50]:
        p_test_N = pos_test[:, :, :N, :]
        v_test_N = vel_test[:, :, :N, :]
        p_init = p_test_N[:, 0]
        v_init = v_test_N[:, 0]
        p_target = p_test_N[:, -1]
        v_target = v_test_N[:, -1]
        energies_gt = []
        for i in range(11):
            e = compute_energy(p_test_N[:, i], v_test_N[:, i], eps=0.01)
            energies_gt.append(e)
        energies_gt = torch.stack(energies_gt, dim=1)
        e_drift_rate_gt = torch.abs(energies_gt[:, -1] - energies_gt[:, 0]).mean().item() / 5.0
        e_mae_gt = torch.abs(energies_gt - energies_gt[:, 0:1]).mean().item()
        with torch.no_grad():
            p_pred_symp, v_pred_symp = solver_symp.integrate_trajectory(p_init, v_init, t_eval)
            p_pred_symp = p_pred_symp.permute(1, 0, 2, 3)
            v_pred_symp = v_pred_symp.permute(1, 0, 2, 3)
            mse_pos_symp = nn.functional.mse_loss(p_pred_symp[:, -1], p_target).item()
            mse_vel_symp = nn.functional.mse_loss(v_pred_symp[:, -1], v_target).item()
            energies_symp = []
            for i in range(11):
                e = compute_energy(p_pred_symp[:, i], v_pred_symp[:, i], eps=gnn_symp.epsilon.item())
                energies_symp.append(e)
            energies_symp = torch.stack(energies_symp, dim=1)
            e_drift_rate_symp = torch.abs(energies_symp[:, -1] - energies_symp[:, 0]).mean().item() / 5.0
            e_mae_symp = torch.abs(energies_symp - energies_symp[:, 0:1]).mean().item()
        with torch.no_grad():
            p_pred_rk4, v_pred_rk4 = solver_rk4.integrate_trajectory(p_init, v_init, t_eval)
            p_pred_rk4 = p_pred_rk4.permute(1, 0, 2, 3)
            v_pred_rk4 = v_pred_rk4.permute(1, 0, 2, 3)
            mse_pos_rk4 = nn.functional.mse_loss(p_pred_rk4[:, -1], p_target).item()
            mse_vel_rk4 = nn.functional.mse_loss(v_pred_rk4[:, -1], v_target).item()
            energies_rk4 = []
            for i in range(11):
                e = compute_energy(p_pred_rk4[:, i], v_pred_rk4[:, i], eps=gnn_rk4.epsilon.item())
                energies_rk4.append(e)
            energies_rk4 = torch.stack(energies_rk4, dim=1)
            e_drift_rate_rk4 = torch.abs(energies_rk4[:, -1] - energies_rk4[:, 0]).mean().item() / 5.0
            e_mae_rk4 = torch.abs(energies_rk4 - energies_rk4[:, 0:1]).mean().item()
        if N == 50:
            with torch.no_grad():
                p_pred_mlp, v_pred_mlp = mlp(p_init, v_init)
                mse_pos_mlp = nn.functional.mse_loss(p_pred_mlp, p_target).item()
                mse_vel_mlp = nn.functional.mse_loss(v_pred_mlp, v_target).item()
                e_init_mlp = compute_energy(p_init, v_init, eps=0.01)
                e_final_mlp = compute_energy(p_pred_mlp, v_pred_mlp, eps=0.01)
                e_drift_rate_mlp = torch.abs(e_final_mlp - e_init_mlp).mean().item() / 5.0
                e_mae_mlp = torch.abs(e_final_mlp - e_init_mlp).mean().item()
        else:
            mse_pos_mlp = None
            mse_vel_mlp = None
            e_drift_rate_mlp = None
            e_mae_mlp = None
        results[N] = {'symp': {'mse_pos': mse_pos_symp, 'mse_vel': mse_vel_symp, 'e_drift_rate': e_drift_rate_symp, 'e_mae': e_mae_symp, 'traj_p': p_pred_symp.cpu(), 'traj_v': v_pred_symp.cpu(), 'energies': energies_symp.cpu()}, 'rk4': {'mse_pos': mse_pos_rk4, 'mse_vel': mse_vel_rk4, 'e_drift_rate': e_drift_rate_rk4, 'e_mae': e_mae_rk4, 'traj_p': p_pred_rk4.cpu(), 'traj_v': v_pred_rk4.cpu(), 'energies': energies_rk4.cpu()}, 'mlp': {'mse_pos': mse_pos_mlp, 'mse_vel': mse_vel_mlp, 'e_drift_rate': e_drift_rate_mlp, 'e_mae': e_mae_mlp}, 'gt': {'traj_p': p_test_N.cpu(), 'traj_v': v_test_N.cpu(), 'energies': energies_gt.cpu()}}
    torch.save(results, 'data/rollout_results.pt')

if __name__ == '__main__':
    evaluate_rollout()
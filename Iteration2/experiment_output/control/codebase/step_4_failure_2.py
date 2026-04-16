# filename: codebase/step_4.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from step_1 import compute_acceleration_batch, compute_energy
from step_2 import InteractionNetwork
from step_3 import compute_energy_pt
def main():
    data_dir = 'data/'
    ic_final_path = os.path.join(data_dir, 'ic_final.npy')
    ic_final = np.load(ic_final_path)
    if ic_final.dtype.names is not None:
        time_field = ic_final['time']
        if time_field.ndim == 2:
            if np.all(time_field[:, 0] == time_field[:, -1]):
                ic_mask = np.isclose(time_field[:, 0], 0.0)
                ic_data = ic_final[ic_mask]
                val_ic_data = ic_data[-20:]
            else:
                ic_mask = np.isclose(time_field[0], 0.0)
                ic_data = ic_final[:, ic_mask]
                val_ic_data = ic_data[-20:]
        else:
            ic_mask = np.isclose(time_field, 0.0)
            ic_data = ic_final[ic_mask]
            val_ic_data = ic_data[-20:]
        val_pos_ic = np.stack([val_ic_data['x'], val_ic_data['y'], val_ic_data['z']], axis=-1)
        val_vel_ic = np.stack([val_ic_data['vx'], val_ic_data['vy'], val_ic_data['vz']], axis=-1)
        val_mass_ic = val_ic_data['mass']
    else:
        if ic_final.ndim == 3:
            time_field = ic_final[:, :, 12]
            if np.all(time_field[:, 0] == time_field[:, -1]):
                ic_mask = np.isclose(time_field[:, 0], 0.0)
                ic_data = ic_final[ic_mask]
                val_ic_data = ic_data[-20:]
            else:
                ic_mask = np.isclose(time_field[0], 0.0)
                ic_data = ic_final[:, ic_mask, :]
                val_ic_data = ic_data[-20:]
            val_pos_ic = val_ic_data[:, :, 0:3]
            val_vel_ic = val_ic_data[:, :, 3:6]
            val_mass_ic = val_ic_data[:, :, 6]
        elif ic_final.ndim == 2:
            time_field = ic_final[:, 12]
            ic_mask = np.isclose(time_field, 0.0)
            ic_data = ic_final[ic_mask]
            ic_data = ic_data.reshape(-1, 50, 13)
            val_ic_data = ic_data[-20:]
            val_pos_ic = val_ic_data[:, :, 0:3]
            val_vel_ic = val_ic_data[:, :, 3:6]
            val_mass_ic = val_ic_data[:, :, 6]
    dt = 0.01
    steps = 500
    epsilon = 0.01
    print('Starting rollouts for 20 held-out simulations (500 steps)...')
    pos_gt = val_pos_ic.copy()
    vel_gt = val_vel_ic.copy()
    mass = val_mass_ic.copy()
    acc_gt = compute_acceleration_batch(pos_gt, mass, epsilon)
    energies_gt = []
    _, _, te = compute_energy(pos_gt, vel_gt, mass, epsilon)
    energies_gt.append(te)
    for step in range(steps):
        vel_half = vel_gt + 0.5 * dt * acc_gt
        pos_gt = pos_gt + dt * vel_half
        acc_gt = compute_acceleration_batch(pos_gt, mass, epsilon)
        vel_gt = vel_half + 0.5 * dt * acc_gt
        _, _, te = compute_energy(pos_gt, vel_gt, mass, epsilon)
        energies_gt.append(te)
    energies_gt = np.array(energies_gt).T
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = InteractionNetwork(hidden_dim=64, epsilon=0.01).to(device)
    model_path = os.path.join(data_dir, 'gnn_model.pth')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    pos_gnn = torch.tensor(val_pos_ic, dtype=torch.float32).to(device)
    vel_gnn = torch.tensor(val_vel_ic, dtype=torch.float32).to(device)
    mass_t = torch.tensor(mass, dtype=torch.float32).to(device)
    with torch.no_grad():
        acc_gnn = model(pos_gnn)
    energies_gnn = []
    _, _, te_gnn = compute_energy_pt(pos_gnn, vel_gnn, mass_t, epsilon)
    energies_gnn.append(te_gnn.cpu().numpy())
    for step in range(steps):
        vel_half = vel_gnn + 0.5 * dt * acc_gnn
        pos_gnn = pos_gnn + dt * vel_half
        with torch.no_grad():
            acc_gnn = model(pos_gnn)
        vel_gnn = vel_half + 0.5 * dt * acc_gnn
        _, _, te_gnn = compute_energy_pt(pos_gnn, vel_gnn, mass_t, epsilon)
        energies_gnn.append(te_gnn.cpu().numpy())
    energies_gnn = np.array(energies_gnn).T
    rel_err_gt = np.abs(energies_gt[:, -1] - energies_gt[:, 0]) / np.abs(energies_gt[:, 0])
    mean_rel_err_gt = np.mean(rel_err_gt)
    std_rel_err_gt = np.std(rel_err_gt)
    rel_err_gnn = np.abs(energies_gnn[:, -1] - energies_gnn[:, 0]) / np.abs(energies_gnn[:, 0])
    mean_rel_err_gnn = np.mean(rel_err_gnn)
    std_rel_err_gnn = np.std(rel_err_gnn)
    pos_mse = np.mean((pos_gnn.cpu().numpy() - pos_gt)**2)
    vel_mse = np.mean((vel_gnn.cpu().numpy() - vel_gt)**2)
    print('--- Rollout Evaluation Results (t=0 to t=5.0) ---')
    print('Ground Truth - Relative Energy Error at t=5.0 | Mean: ' + str(mean_rel_err_gt) + ' | Std: ' + str(std_rel_err_gt))
    print('GNN - Relative Energy Error at t=5.0 | Mean: ' + str(mean_rel_err_gnn) + ' | Std: ' + str(std_rel_err_gnn))
    print('Final Position MSE (GNN vs GT): ' + str(pos_mse))
    print('Final Velocity MSE (GNN vs GT): ' + str(vel_mse))
    fig, axes = plt.subplots(4, 5, figsize=(20, 16))
    time_axis = np.linspace(0, 5.0, steps + 1)
    for i in range(20):
        ax = axes[i // 5, i % 5]
        ax.plot(time_axis, energies_gt[i], label='Ground Truth', color='black', linestyle='--')
        ax.plot(time_axis, energies_gnn[i], label='GNN', color='red', alpha=0.7)
        ax.set_title('Sim ' + str(80 + i))
        initial_energy = energies_gt[i][0]
        y_min = initial_energy - 0.05 * np.abs(initial_energy)
        y_max = initial_energy + 0.05 * np.abs(initial_energy)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel('Time (T)')
        ax.set_ylabel('Total Energy')
        ax.legend(fontsize='small', loc='upper left')
    plt.tight_layout()
    timestamp = int(time.time())
    plot_filename = os.path.join(data_dir, 'energy_evolution_4_' + str(timestamp) + '.png')
    plt.savefig(plot_filename, dpi=300)
    print('Energy evolution plot saved to ' + plot_filename)
if __name__ == '__main__':
    main()
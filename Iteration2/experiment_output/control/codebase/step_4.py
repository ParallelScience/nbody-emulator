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
    dataset_path = os.path.join(data_dir, 'processed_trajectory_dataset.npz')
    dataset = np.load(dataset_path)
    pos_all = dataset['pos']
    vel_all = dataset['vel']
    mass_all = dataset['mass']
    val_pos_ic = pos_all[80:, 0, :, :]
    val_vel_ic = vel_all[80:, 0, :, :]
    val_mass_ic = mass_all[80:, 0, :]
    dt = 0.01
    steps = 500
    epsilon = 0.01
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
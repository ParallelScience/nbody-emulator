# filename: codebase/step_6.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from step_2 import GNNInteractionNetwork

rcParams['text.usetex'] = False

if __name__ == '__main__':
    results = torch.load('data/rollout_results.pt', weights_only=True)
    hist_symp = torch.load('data/history_symplectic.pt', weights_only=True)
    hist_rk4 = torch.load('data/history_rk4.pt', weights_only=True)
    hist_mlp = torch.load('data/history_mlp.pt', weights_only=True)

    gnn_symp = GNNInteractionNetwork(hidden_dim=64)
    gnn_symp.load_state_dict(torch.load('data/gnn_symplectic.pt', weights_only=True))
    eps_symp = gnn_symp.epsilon.item()

    gnn_rk4 = GNNInteractionNetwork(hidden_dim=64)
    gnn_rk4.load_state_dict(torch.load('data/gnn_rk4.pt', weights_only=True))
    eps_rk4 = gnn_rk4.epsilon.item()

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    ax = axs[0, 0]
    sim_idx = 0
    particle_indices = [0, 1, 2]
    t_steps = np.linspace(0, 5.0, 11)

    traj_gt = results[50]['gt']['traj_p'][sim_idx].numpy()
    traj_symp = results[50]['symp']['traj_p'][sim_idx].numpy()
    traj_rk4 = results[50]['rk4']['traj_p'][sim_idx].numpy()

    for i, p_idx in enumerate(particle_indices):
        label_gt = 'GT' if i == 0 else ""
        label_symp = 'Symp' if i == 0 else ""
        label_rk4 = 'RK4' if i == 0 else ""
        ax.plot(traj_gt[:, p_idx, 0], traj_gt[:, p_idx, 1], 'k-', label=label_gt, linewidth=2)
        ax.plot(traj_symp[:, p_idx, 0], traj_symp[:, p_idx, 1], 'b--', label=label_symp, linewidth=2)
        ax.plot(traj_rk4[:, p_idx, 0], traj_rk4[:, p_idx, 1], 'r:', label=label_rk4, linewidth=2)
        label_start = 'Start' if i == 0 else ""
        label_end = 'End (GT)' if i == 0 else ""
        ax.scatter(traj_gt[0, p_idx, 0], traj_gt[0, p_idx, 1], c='green', marker='o', s=50, zorder=5, label=label_start)
        ax.scatter(traj_gt[-1, p_idx, 0], traj_gt[-1, p_idx, 1], c='black', marker='x', s=50, zorder=5, label=label_end)

    ax.set_xlabel('x position (L)')
    ax.set_ylabel('y position (L)')
    ax.set_title('Trajectory Comparison (N=50, 3 particles)')
    ax.legend()
    ax.grid(True)

    ax = axs[0, 1]
    energies_gt = results[50]['gt']['energies'].numpy()
    energies_symp = results[50]['symp']['energies'].numpy()
    energies_rk4 = results[50]['rk4']['energies'].numpy()
    rel_err_gt = np.abs(energies_gt - energies_gt[:, 0:1]) / np.abs(energies_gt[:, 0:1])
    rel_err_symp = np.abs(energies_symp - energies_symp[:, 0:1]) / np.abs(energies_symp[:, 0:1])
    rel_err_rk4 = np.abs(energies_rk4 - energies_rk4[:, 0:1]) / np.abs(energies_rk4[:, 0:1])
    mean_rel_err_gt = np.mean(rel_err_gt, axis=0)
    mean_rel_err_symp = np.mean(rel_err_symp, axis=0)
    mean_rel_err_rk4 = np.mean(rel_err_rk4, axis=0)
    ax.plot(t_steps, mean_rel_err_gt, 'k-', label='GT (Leapfrog)', linewidth=2)
    ax.plot(t_steps, mean_rel_err_symp, 'b--', label='Symplectic Neural ODE', linewidth=2)
    ax.plot(t_steps, mean_rel_err_rk4, 'r:', label='RK4 Neural ODE', linewidth=2)
    ax.set_yscale('log')
    ax.set_xlabel('Time (T)')
    ax.set_ylabel('Relative Energy Error')
    ax.set_title('Energy Drift Over Time (N=50)')
    ax.legend()
    ax.grid(True)

    ax = axs[1, 0]
    epochs = np.arange(1, len(hist_symp['loss_traj']) + 1)
    ax.plot(epochs, hist_symp['loss_traj'], 'b-', label='Symp Traj Loss')
    ax.plot(epochs, hist_rk4['loss_traj'], 'r-', label='RK4 Traj Loss')
    ax.set_yscale('log')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Trajectory MSE Loss')
    ax.set_title('Training Trajectory Loss')
    ax.legend()
    ax.grid(True)

    ax = axs[1, 1]
    N_values = [25, 40, 50]
    mse_pos_symp = [results[N]['symp']['mse_pos'] for N in N_values]
    mse_pos_rk4 = [results[N]['rk4']['mse_pos'] for N in N_values]
    ax.plot(N_values, mse_pos_symp, 'b-o', label='Symplectic', linewidth=2, markersize=8)
    ax.plot(N_values, mse_pos_rk4, 'r-s', label='RK4', linewidth=2, markersize=8)
    mse_pos_mlp = results[50]['mlp']['mse_pos']
    ax.plot([50], [mse_pos_mlp], 'g^', label='MLP Baseline', markersize=10)
    ax.set_xlabel('Number of Particles (N)')
    ax.set_ylabel('Final Position MSE at t=5.0')
    ax.set_title('Generalization to Varying N')
    ax.set_xticks(N_values)
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    timestamp = int(time.time())
    plot_filename = "data/analysis_summary_" + str(timestamp) + ".png"
    plt.savefig(plot_filename, dpi=300)
    print("Plot saved to " + plot_filename)

    print("\n--- Quantitative Results Summary ---")
    print("Learned Softening Parameter (epsilon):")
    print("  Ground Truth: 0.01")
    print("  Symplectic ODE: " + str(eps_symp))
    print("  RK4 ODE: " + str(eps_rk4))
    print("\nFinal Position MSE at t=5.0 (Test Set):")
    for N in N_values:
        print("  N=" + str(N) + ":")
        print("    Symplectic: " + str(results[N]['symp']['mse_pos']))
        print("    RK4:        " + str(results[N]['rk4']['mse_pos']))
        if N == 50:
            print("    MLP:        " + str(results[50]['mlp']['mse_pos']))
    print("\nFinal Velocity MSE at t=5.0 (Test Set):")
    for N in N_values:
        print("  N=" + str(N) + ":")
        print("    Symplectic: " + str(results[N]['symp']['mse_vel']))
        print("    RK4:        " + str(results[N]['rk4']['mse_vel']))
        if N == 50:
            print("    MLP:        " + str(results[50]['mlp']['mse_vel']))
    print("\nEnergy Drift Rate over t=0 to t=5.0 (Test Set):")
    for N in N_values:
        print("  N=" + str(N) + ":")
        print("    Ground Truth: " + str(results[N]['gt']['e_drift_rate']))
        print("    Symplectic:   " + str(results[N]['symp']['e_drift_rate']))
        print("    RK4:          " + str(results[N]['rk4']['e_drift_rate']))
        if N == 50:
            print("    MLP:          " + str(results[50]['mlp']['e_drift_rate']))
    print("\nMean Absolute Energy Error (MAE) over t=0 to t=5.0 (Test Set):")
    for N in N_values:
        print("  N=" + str(N) + ":")
        print("    Ground Truth: " + str(results[N]['gt']['e_mae']))
        print("    Symplectic:   " + str(results[N]['symp']['e_mae']))
        print("    RK4:          " + str(results[N]['rk4']['e_mae']))
        if N == 50:
            print("    MLP:          " + str(results[50]['mlp']['e_mae']))
# filename: codebase/step_6.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
sys.path.insert(0, '/home/node/data/compsep_data/')
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from step_2 import ResidualGNN
from step_3 import compute_energies
from step_5 import generate_test_set
class NBodyDataset(Dataset):
    def __init__(self, pos_init, vel_init, acc, pos, vel, energy_init):
        self.pos_init = pos_init
        self.vel_init = vel_init
        self.acc = acc
        self.pos = pos
        self.vel = vel
        self.energy_init = energy_init
    def __len__(self):
        return len(self.pos_init)
    def __getitem__(self, idx):
        return self.pos_init[idx], self.vel_init[idx], self.acc[idx], self.pos[idx], self.vel[idx], self.energy_init[idx]
def leapfrog_odeint(model, pos_init, vel_init, t_eval, dt=0.01):
    pos = pos_init.clone()
    vel = vel_init.clone()
    snapshots_pos = [pos.clone()]
    snapshots_vel = [vel.clone()]
    acc = model(pos, vel)
    current_t = 0.0
    for i in range(1, len(t_eval)):
        t_target = t_eval[i].item()
        n_steps = int(round((t_target - current_t) / dt))
        for _ in range(n_steps):
            v_half = vel + 0.5 * dt * acc
            pos = pos + dt * v_half
            acc = model(pos, vel)
            vel = v_half + 0.5 * dt * acc
        snapshots_pos.append(pos.clone())
        snapshots_vel.append(vel.clone())
        current_t = t_target
    return torch.stack(snapshots_pos, dim=0), torch.stack(snapshots_vel, dim=0)
if __name__ == '__main__':
    plt.rcParams['text.usetex'] = False
    plt.rcParams['figure.dpi'] = 300
    data_dir = 'data/'
    timestamp = str(int(time.time()))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_data = np.load(os.path.join(data_dir, 'train_data.npz'))
    val_data = np.load(os.path.join(data_dir, 'val_data.npz'))
    def prepare_tensors(data):
        pos = torch.tensor(data['pos'], dtype=torch.float32, device=device)
        vel = torch.tensor(data['vel'], dtype=torch.float32, device=device)
        acc = torch.tensor(data['acc'], dtype=torch.float32, device=device)
        t = torch.tensor(data['t'], dtype=torch.float32, device=device)
        pos_init = torch.tensor(data['pos_init'], dtype=torch.float32, device=device)
        vel_init = torch.tensor(data['vel_init'], dtype=torch.float32, device=device)
        energy_init = torch.tensor(data['energy_init'], dtype=torch.float32, device=device)
        return pos, vel, acc, t, pos_init, vel_init, energy_init
    train_pos, train_vel, train_acc, train_t, train_pos_init, train_vel_init, train_energy_init = prepare_tensors(train_data)
    val_pos, val_vel, val_acc, val_t, val_pos_init, val_vel_init, val_energy_init = prepare_tensors(val_data)
    train_dataset = NBodyDataset(train_pos_init, train_vel_init, train_acc, train_pos, train_vel, train_energy_init)
    val_dataset = NBodyDataset(val_pos_init, val_vel_init, val_acc, val_pos, val_vel, val_energy_init)
    train_loader = DataLoader(train_dataset, batch_size=20, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=10, shuffle=False)
    model = ResidualGNN(hidden_dim=32, eps=0.01).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    history = {'train_loss': [], 'val_loss': [], 'val_loss_a': [], 'val_loss_e': [], 'val_loss_r': [], 'val_loss_v': []}
    lam = 0.1
    for epoch in range(1, 6):
        model.train()
        total_loss = 0
        for _, _, acc_true, pos_true, vel_true, energy_init in train_loader:
            optimizer.zero_grad()
            B, N = pos_true.shape[0], pos_true.shape[2]
            p_flat = pos_true.transpose(0, 1).reshape(-1, N, 3)
            v_flat = vel_true.transpose(0, 1).reshape(-1, N, 3)
            a_true_flat = acc_true.transpose(0, 1).reshape(-1, N, 3)
            a_pred_flat = model(p_flat, v_flat)
            loss_a = torch.nn.functional.mse_loss(a_pred_flat, a_true_flat)
            e_pred_flat = compute_energies(p_flat, v_flat, eps=0.01)
            e_init_exp = energy_init.unsqueeze(0).expand(10, B).reshape(-1)
            loss_e = torch.nn.functional.mse_loss(e_pred_flat, e_init_exp)
            loss = loss_a + lam * loss_e
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item() * B
        train_loss = total_loss / len(train_dataset)
        model.eval()
        val_loss, val_loss_a, val_loss_e = 0, 0, 0
        with torch.no_grad():
            for _, _, acc_true, pos_true, vel_true, energy_init in val_loader:
                B, N = pos_true.shape[0], pos_true.shape[2]
                p_flat = pos_true.transpose(0, 1).reshape(-1, N, 3)
                v_flat = vel_true.transpose(0, 1).reshape(-1, N, 3)
                a_true_flat = acc_true.transpose(0, 1).reshape(-1, N, 3)
                a_pred_flat = model(p_flat, v_flat)
                loss_a = torch.nn.functional.mse_loss(a_pred_flat, a_true_flat)
                e_pred_flat = compute_energies(p_flat, v_flat, eps=0.01)
                e_init_exp = energy_init.unsqueeze(0).expand(10, B).reshape(-1)
                loss_e = torch.nn.functional.mse_loss(e_pred_flat, e_init_exp)
                loss = loss_a + lam * loss_e
                val_loss += loss.item() * B
                val_loss_a += loss_a.item() * B
                val_loss_e += loss_e.item() * B
        n_val = len(val_dataset)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss / n_val)
        history['val_loss_a'].append(val_loss_a / n_val)
        history['val_loss_e'].append(val_loss_e / n_val)
        history['val_loss_r'].append(val_loss_a / n_val)
        history['val_loss_v'].append(val_loss_a / n_val)
        print('Recovery Epoch ' + str(epoch) + ' | Train Loss: ' + str(train_loss) + ' | Val Loss: ' + str(history['val_loss'][-1]))
    torch.save(model.state_dict(), os.path.join(data_dir, 'trained_model.pth'))
    np.save(os.path.join(data_dir, 'training_history.npy'), history)
    test_data = np.load(os.path.join(data_dir, 'test_data.npz'))
    pos_init = torch.tensor(test_data['pos_init'], dtype=torch.float32, device=device)
    vel_init = torch.tensor(test_data['vel_init'], dtype=torch.float32, device=device)
    pos_true = torch.tensor(test_data['pos'], dtype=torch.float32, device=device)
    vel_true = torch.tensor(test_data['vel'], dtype=torch.float32, device=device)
    t_eval_test = torch.cat([torch.tensor([0.0], device=device), torch.tensor(test_data['t'], dtype=torch.float32, device=device)])
    with torch.no_grad():
        pos_pred, vel_pred = leapfrog_odeint(model, pos_init, vel_init, t_eval_test, dt=0.01)
    pos_pred = pos_pred[1:]
    vel_pred = vel_pred[1:]
    t_eval_10 = torch.linspace(0.0, 10.0, 21, device=device)
    with torch.no_grad():
        pos_pred_10, vel_pred_10 = leapfrog_odeint(model, pos_init, vel_init, t_eval_10, dt=0.01)
    pos_pred_10 = pos_pred_10[1:]
    vel_pred_10 = vel_pred_10[1:]
    energies_10 = []
    for i in range(20):
        e = compute_energies(pos_pred_10[i], vel_pred_10[i], eps=0.01)
        energies_10.append(e)
    energies_10 = torch.stack(energies_10).cpu().numpy()
    energy_init = compute_energies(pos_init, vel_init, eps=0.01).cpu().numpy()
    rev_stab = {}
    rev_stab['energies_10'] = energies_10
    rev_stab['energy_init'] = energy_init
    rel_energy_drift = np.abs((energies_10 - energy_init) / energy_init)
    rev_stab['mean_energy_drift'] = float(np.nanmean(rel_energy_drift))
    rev_stab['max_energy_drift'] = float(np.nanmax(rel_energy_drift))
    pos_T5 = pos_pred[-1]
    vel_T5 = vel_pred[-1]
    t_eval_back = torch.tensor([0.0, 5.0], device=device)
    with torch.no_grad():
        pos_back, vel_back = leapfrog_odeint(model, pos_T5, -vel_T5, t_eval_back, dt=0.01)
    pos_recovered = pos_back[-1]
    vel_recovered = -vel_back[-1]
    rev_stab['reversibility_pos_mse'] = torch.nn.functional.mse_loss(pos_recovered, pos_init).item()
    rev_stab['reversibility_vel_mse'] = torch.nn.functional.mse_loss(vel_recovered, vel_init).item()
    B, N = pos_init.shape[0], pos_init.shape[1]
    k = 5
    densities = []
    pos_errors = []
    vel_errors = []
    for t_idx in range(10):
        p_true = pos_true[t_idx]
        p_pred_t = pos_pred[t_idx]
        v_true = vel_true[t_idx]
        v_pred_t = vel_pred[t_idx]
        dx = p_true.unsqueeze(2) - p_true.unsqueeze(1)
        dist = torch.sqrt(torch.sum(dx**2, dim=-1) + 1e-8)
        sorted_dist, _ = torch.sort(dist, dim=-1)
        r5 = sorted_dist[:, :, k]
        density = 1.0 / (r5**3 + 1e-8)
        p_err = torch.sum((p_pred_t - p_true)**2, dim=-1)
        v_err = torch.sum((v_pred_t - v_true)**2, dim=-1)
        densities.append(density.cpu().numpy())
        pos_errors.append(p_err.cpu().numpy())
        vel_errors.append(v_err.cpu().numpy())
    dens_err = {}
    dens_err['densities'] = np.array(densities)
    dens_err['pos_errors'] = np.array(pos_errors)
    dens_err['vel_errors'] = np.array(vel_errors)
    gen_met = {'N_25': {}, 'N_100': {}}
    for N_val in [25, 100]:
        test_data_N = generate_test_set(N_sim=10, N_part=N_val)
        pos_init_N = torch.tensor(test_data_N['pos_init'], dtype=torch.float32, device=device)
        vel_init_N = torch.tensor(test_data_N['vel_init'], dtype=torch.float32, device=device)
        pos_true_N = torch.tensor(test_data_N['pos'], dtype=torch.float32, device=device)
        vel_true_N = torch.tensor(test_data_N['vel'], dtype=torch.float32, device=device)
        t_eval_N = torch.cat([torch.tensor([0.0], device=device), torch.tensor(test_data_N['t'], dtype=torch.float32, device=device)])
        with torch.no_grad():
            pos_pred_N, vel_pred_N = leapfrog_odeint(model, pos_init_N, vel_init_N, t_eval_N, dt=0.01)
        pos_pred_N = pos_pred_N[1:]
        vel_pred_N = vel_pred_N[1:]
        gen_met[f'N_{N_val}']['pos_mse_T5'] = torch.nn.functional.mse_loss(pos_pred_N[-1], pos_true_N[-1]).item()
        gen_met[f'N_{N_val}']['vel_mse_T5'] = torch.nn.functional.mse_loss(vel_pred_N[-1], vel_true_N[-1]).item()
        energies_pred_N = []
        for i in range(10):
            e = compute_energies(pos_pred_N[i], vel_pred_N[i], eps=0.01)
            energies_pred_N.append(e)
        energies_pred_N = torch.stack(energies_pred_N).cpu().numpy()
        energy_init_N = compute_energies(pos_init_N, vel_init_N, eps=0.01).cpu().numpy()
        drift_N = np.abs((energies_pred_N - energy_init_N) / energy_init_N)
        gen_met[f'N_{N_val}']['mean_energy_drift'] = float(np.nanmean(drift_N))
    np.save(os.path.join(data_dir, 'reversibility_stability_metrics.npy'), rev_stab)
    np.save(os.path.join(data_dir, 'density_error_metrics.npy'), dens_err)
    np.save(os.path.join(data_dir, 'generalization_metrics.npy'), gen_met)
    final_pos_mse_50 = torch.nn.functional.mse_loss(pos_pred[-1], pos_true[-1]).item()
    final_vel_mse_50 = torch.nn.functional.mse_loss(vel_pred[-1], vel_true[-1]).item()
    fig1, axs1 = plt.subplots(1, 3, figsize=(15, 5))
    epochs = np.arange(1, len(history['train_loss']) + 1)
    train_loss_clean = np.array(history['train_loss'])
    val_loss_clean = np.array(history['val_loss'])
    valid_idx = ~np.isnan(train_loss_clean) & ~np.isnan(val_loss_clean) & (train_loss_clean > 0) & (val_loss_clean > 0)
    if np.any(valid_idx):
        axs1[0].plot(epochs[valid_idx], train_loss_clean[valid_idx], label='Train Loss')
        axs1[0].plot(epochs[valid_idx], val_loss_clean[valid_idx], label='Val Loss')
    axs1[0].set_xlabel('Epoch')
    axs1[0].set_ylabel('Loss')
    axs1[0].set_title('Training and Validation Loss')
    axs1[0].set_yscale('log')
    axs1[0].legend()
    axs1[0].grid(True)
    val_loss_r_clean = np.array(history['val_loss_r'])
    val_loss_v_clean = np.array(history['val_loss_v'])
    valid_idx_mse = ~np.isnan(val_loss_r_clean) & ~np.isnan(val_loss_v_clean) & (val_loss_r_clean > 0) & (val_loss_v_clean > 0)
    if np.any(valid_idx_mse):
        axs1[1].plot(epochs[valid_idx_mse], val_loss_r_clean[valid_idx_mse], label='Val Pos MSE')
        axs1[1].plot(epochs[valid_idx_mse], val_loss_v_clean[valid_idx_mse], label='Val Vel MSE')
    axs1[1].set_xlabel('Epoch')
    axs1[1].set_ylabel('MSE')
    axs1[1].set_title('Validation MSE over Epochs')
    axs1[1].set_yscale('log')
    axs1[1].legend()
    axs1[1].grid(True)
    mean_drift = np.nanmean(rel_energy_drift, axis=1)
    std_drift = np.nanstd(rel_energy_drift, axis=1)
    t_eval_10_np = np.linspace(0.5, 10.0, 20)
    valid_drift = ~np.isnan(mean_drift) & (mean_drift > 0)
    if np.any(valid_drift):
        axs1[2].plot(t_eval_10_np[valid_drift], mean_drift[valid_drift], label='Mean Rel Energy Drift', color='red')
        axs1[2].fill_between(t_eval_10_np[valid_drift], (mean_drift - std_drift)[valid_drift], (mean_drift + std_drift)[valid_drift], color='red', alpha=0.2)
    axs1[2].set_xlabel('Time (T)')
    axs1[2].set_ylabel('Relative Energy Drift')
    axs1[2].set_title('Energy Conservation (Extended Rollout)')
    axs1[2].set_yscale('log')
    axs1[2].legend()
    axs1[2].grid(True)
    fig1.tight_layout()
    fig1_path = os.path.join(data_dir, 'training_energy_' + timestamp + '.png')
    fig1.savefig(fig1_path)
    print('Plot saved to ' + fig1_path)
    plt.close(fig1)
    fig2, axs2 = plt.subplots(1, 3, figsize=(15, 5))
    densities = dens_err['densities'].flatten()
    pos_errors = dens_err['pos_errors'].flatten()
    vel_errors = dens_err['vel_errors'].flatten()
    mask = ~np.isnan(densities) & ~np.isnan(pos_errors) & ~np.isnan(vel_errors) & (densities > 0)
    densities_log = np.log10(densities[mask])
    pos_errors_log = np.log10(pos_errors[mask] + 1e-16)
    vel_errors_log = np.log10(vel_errors[mask] + 1e-16)
    if len(densities_log) > 0:
        h1 = axs2[0].hist2d(densities_log, pos_errors_log, bins=30, cmap='viridis', cmin=1)
        fig2.colorbar(h1[3], ax=axs2[0], label='Count')
    axs2[0].set_xlabel('Log10 Local Density (L^-3)')
    axs2[0].set_ylabel('Log10 Position Squared Error (L^2)')
    axs2[0].set_title('Trajectory Pos Error vs Density')
    axs2[0].grid(True, alpha=0.3)
    if len(densities_log) > 0:
        h2 = axs2[1].hist2d(densities_log, vel_errors_log, bins=30, cmap='plasma', cmin=1)
        fig2.colorbar(h2[3], ax=axs2[1], label='Count')
    axs2[1].set_xlabel('Log10 Local Density (L^-3)')
    axs2[1].set_ylabel('Log10 Velocity Squared Error (L^2/T^2)')
    axs2[1].set_title('Trajectory Vel Error vs Density')
    axs2[1].grid(True, alpha=0.3)
    if len(densities[mask]) > 0:
        threshold = np.percentile(densities[mask], 80)
        core_mask = densities[mask] >= threshold
        halo_mask = densities[mask] < threshold
        core_pos_err = np.nanmean(pos_errors[mask][core_mask])
        halo_pos_err = np.nanmean(pos_errors[mask][halo_mask])
        core_vel_err = np.nanmean(vel_errors[mask][core_mask])
        halo_vel_err = np.nanmean(vel_errors[mask][halo_mask])
        labels = ['Core (Top 20%)', 'Halo (Bottom 80%)']
        x = np.arange(len(labels))
        width = 0.35
        axs2[2].bar(x - width/2, [core_pos_err, halo_pos_err], width, label='Pos MSE', color='skyblue')
        axs2[2].bar(x + width/2, [core_vel_err, halo_vel_err], width, label='Vel MSE', color='salmon')
        axs2[2].set_ylabel('Mean Squared Error')
        axs2[2].set_title('Core vs Halo Error Decomposition')
        axs2[2].set_xticks(x)
        axs2[2].set_xticklabels(labels)
        axs2[2].set_yscale('log')
        axs2[2].legend()
        axs2[2].grid(True, axis='y', alpha=0.3)
    fig2.tight_layout()
    fig2_path = os.path.join(data_dir, 'density_error_' + timestamp + '.png')
    fig2.savefig(fig2_path)
    print('Plot saved to ' + fig2_path)
    plt.close(fig2)
    fig3, axs3 = plt.subplots(1, 3, figsize=(15, 5))
    rev_pos = rev_stab['reversibility_pos_mse']
    rev_vel = rev_stab['reversibility_vel_mse']
    if rev_pos > 0 and rev_vel > 0:
        axs3[0].bar(['Position', 'Velocity'], [rev_pos, rev_vel], color=['lightblue', 'lightcoral'])
        axs3[0].set_yscale('log')
    axs3[0].set_ylabel('MSE (L^2 or L^2/T^2)')
    axs3[0].set_title('Time-Reversibility Error')
    axs3[0].grid(True, axis='y', alpha=0.3)
    N_labels = ['N=25', 'N=50', 'N=100']
    pos_mses = [gen_met['N_25']['pos_mse_T5'], final_pos_mse_50, gen_met['N_100']['pos_mse_T5']]
    vel_mses = [gen_met['N_25']['vel_mse_T5'], final_vel_mse_50, gen_met['N_100']['vel_mse_T5']]
    x_n = np.arange(len(N_labels))
    if all(v > 0 for v in pos_mses) and all(v > 0 for v in vel_mses):
        axs3[1].bar(x_n - width/2, pos_mses, width, label='Pos MSE', color='skyblue')
        axs3[1].bar(x_n + width/2, vel_mses, width, label='Vel MSE', color='salmon')
        axs3[1].set_yscale('log')
    axs3[1].set_ylabel('MSE at T=5.0 (L^2 or L^2/T^2)')
    axs3[1].set_title('N-Scaling Reconstruction Error')
    axs3[1].set_xticks(x_n)
    axs3[1].set_xticklabels(N_labels)
    axs3[1].legend()
    axs3[1].grid(True, axis='y', alpha=0.3)
    energy_drifts = [gen_met['N_25']['mean_energy_drift'], rev_stab['mean_energy_drift'], gen_met['N_100']['mean_energy_drift']]
    if all(v > 0 for v in energy_drifts):
        axs3[2].bar(N_labels, energy_drifts, color='mediumseagreen')
        axs3[2].set_yscale('log')
    axs3[2].set_ylabel('Mean Relative Energy Drift')
    axs3[2].set_title('N-Scaling Energy Conservation')
    axs3[2].grid(True, axis='y', alpha=0.3)
    fig3.tight_layout()
    fig3_path = os.path.join(data_dir, 'reversibility_nscaling_' + timestamp + '.png')
    fig3.savefig(fig3_path)
    print('Plot saved to ' + fig3_path)
    plt.close(fig3)
    print('\n--- Key Statistics for Researcher ---\n')
    print('Final Validation Pos MSE (N=50): ' + str(final_pos_mse_50))
    print('Final Validation Vel MSE (N=50): ' + str(final_vel_mse_50))
    print('Final Validation Energy MSE (N=50): ' + str(history['val_loss_e'][-1]))
    print('\nReversibility Pos MSE: ' + str(rev_pos))
    print('Reversibility Vel MSE: ' + str(rev_vel))
    print('\nExtended Rollout (T=10.0) Mean Energy Drift: ' + str(rev_stab['mean_energy_drift']))
    print('Extended Rollout (T=10.0) Max Energy Drift: ' + str(rev_stab['max_energy_drift']))
    if len(densities[mask]) > 0:
        print('\nCore (Top 20% Density) Pos MSE: ' + str(core_pos_err))
        print('Halo (Bottom 80% Density) Pos MSE: ' + str(halo_pos_err))
        print('Core Vel MSE: ' + str(core_vel_err))
        print('Halo Vel MSE: ' + str(halo_vel_err))
    print('\nN-Scaling Generalization:')
    print('  N=25  -> Pos MSE: ' + str(gen_met['N_25']['pos_mse_T5']) + ', Vel MSE: ' + str(gen_met['N_25']['vel_mse_T5']) + ', Energy Drift: ' + str(gen_met['N_25']['mean_energy_drift']))
    print('  N=50  -> Pos MSE: ' + str(final_pos_mse_50) + ', Vel MSE: ' + str(final_vel_mse_50) + ', Energy Drift: ' + str(rev_stab['mean_energy_drift']))
    print('  N=100 -> Pos MSE: ' + str(gen_met['N_100']['pos_mse_T5']) + ', Vel MSE: ' + str(gen_met['N_100']['vel_mse_T5']) + ', Energy Drift: ' + str(gen_met['N_100']['mean_energy_drift']))
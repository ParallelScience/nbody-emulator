# filename: codebase/step_3.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.checkpoint as checkpoint
import matplotlib.pyplot as plt
import time
from step_2 import InteractionNetwork
def compute_energy_batch(pos, vel, eps=0.01):
    KE = 0.5 * torch.sum(vel**2, dim=(1, 2))
    batch_size, N, _ = pos.shape
    i, j = torch.triu_indices(N, N, offset=1, device=pos.device)
    r_ij = pos[:, j, :] - pos[:, i, :]
    dist_sq = torch.sum(r_ij**2, dim=-1) + eps**2
    dist = torch.sqrt(dist_sq)
    PE = -torch.sum(1.0 / dist, dim=1)
    return KE + PE
def augment_trajectory(traj):
    batch_size = traj.shape[0]
    aug_traj = traj.clone()
    for b in range(batch_size):
        H = torch.randn(3, 3)
        Q, R_mat = torch.linalg.qr(H)
        signs = torch.sign(torch.diag(R_mat))
        signs[signs == 0] = 1
        Q = Q * signs
        if torch.linalg.det(Q) < 0:
            Q[:, 0] *= -1
        reflect = torch.diag(torch.randint(0, 2, (3,), dtype=torch.float32) * 2 - 1)
        transformation = Q @ reflect
        pos = traj[b, :, :, :3]
        vel = traj[b, :, :, 3:]
        aug_pos = torch.matmul(pos, transformation.T)
        aug_vel = torch.matmul(vel, transformation.T)
        aug_traj[b, :, :, :3] = aug_pos
        aug_traj[b, :, :, 3:] = aug_vel
    return aug_traj
class NeuralLeapfrog(nn.Module):
    def __init__(self, network, steps_per_snapshot=50, num_snapshots=10, dt=0.01):
        super().__init__()
        self.network = network
        self.steps_per_snapshot = steps_per_snapshot
        self.num_snapshots = num_snapshots
        self.dt = dt
    def forward(self, pos_0, vel_0, R, V):
        dt_norm = self.dt * V / R
        pos = pos_0
        vel = vel_0
        snapshots = []
        def get_acc(p):
            if p.requires_grad:
                return checkpoint.checkpoint(self.network, p, use_reentrant=False)
            else:
                return self.network(p)
        acc = get_acc(pos)
        for i in range(self.num_snapshots):
            for _ in range(self.steps_per_snapshot):
                vel = vel + 0.5 * dt_norm * acc
                pos = pos + dt_norm * vel
                acc = get_acc(pos)
                vel = vel + 0.5 * dt_norm * acc
            snapshots.append(torch.cat([pos, vel], dim=-1))
        return torch.stack(snapshots, dim=1)
if __name__ == '__main__':
    torch.manual_seed(42)
    np.random.seed(42)
    data_dir = 'data/'
    trajectories = np.load(os.path.join(data_dir, 'trajectories.npy'))
    norm_coeffs = np.load(os.path.join(data_dir, 'normalization_coeffs.npy'))
    R = norm_coeffs[:, 0]
    V = norm_coeffs[:, 1]
    traj_norm = np.zeros_like(trajectories)
    traj_norm[:, :, :, :3] = trajectories[:, :, :, :3] / R[:, None, None, None]
    traj_norm[:, :, :, 3:] = trajectories[:, :, :, 3:] / V[:, None, None, None]
    traj_norm = torch.tensor(traj_norm, dtype=torch.float32)
    R_tensor = torch.tensor(R, dtype=torch.float32).view(-1, 1, 1)
    V_tensor = torch.tensor(V, dtype=torch.float32).view(-1, 1, 1)
    train_traj = traj_norm[:80]
    val_traj = traj_norm[80:]
    train_R = R_tensor[:80]
    train_V = V_tensor[:80]
    val_R = R_tensor[80:]
    val_V = V_tensor[80:]
    epochs = 100
    hidden_dim = 64
    lr = 1e-3
    noise_std = 0.01
    network = InteractionNetwork(hidden_dim=hidden_dim, use_softening=True)
    model = NeuralLeapfrog(network, steps_per_snapshot=50, num_snapshots=10, dt=0.01)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.5)
    train_losses, val_losses, train_mse_losses, val_mse_losses, train_energy_losses, val_energy_losses = [], [], [], [], [], []
    start_time = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        aug_train_traj = augment_trajectory(train_traj)
        pos_0 = aug_train_traj[:, 0, :, :3]
        vel_0 = aug_train_traj[:, 0, :, 3:]
        true_snapshots = aug_train_traj[:, 1:, :, :]
        pos_0_noisy = pos_0 + torch.randn_like(pos_0) * noise_std
        vel_0_noisy = vel_0 + torch.randn_like(vel_0) * noise_std
        pos_0_noisy.requires_grad_()
        pos_0_noisy_phys = pos_0_noisy * train_R
        vel_0_noisy_phys = vel_0_noisy * train_V
        E_init_noisy = compute_energy_batch(pos_0_noisy_phys, vel_0_noisy_phys)
        pred_snapshots = model(pos_0_noisy, vel_0_noisy, train_R, train_V)
        lambda_E = min(1.0, epoch / 50.0) * 1.0
        mse_loss = nn.functional.mse_loss(pred_snapshots, true_snapshots)
        pred_pos_phys = pred_snapshots[:, :, :, :3] * train_R.unsqueeze(1)
        pred_vel_phys = pred_snapshots[:, :, :, 3:] * train_V.unsqueeze(1)
        energy_loss = 0.0
        for i in range(pred_snapshots.shape[1]):
            E_pred = compute_energy_batch(pred_pos_phys[:, i], pred_vel_phys[:, i])
            rel_energy_error = torch.abs(E_pred - E_init_noisy) / torch.abs(E_init_noisy)
            energy_loss += torch.mean(rel_energy_error)
        energy_loss = energy_loss / pred_snapshots.shape[1]
        loss = mse_loss + lambda_E * energy_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        model.eval()
        with torch.no_grad():
            val_pos_0 = val_traj[:, 0, :, :3]
            val_vel_0 = val_traj[:, 0, :, 3:]
            val_true_snapshots = val_traj[:, 1:, :, :]
            val_pos_0_phys = val_pos_0 * val_R
            val_vel_0_phys = val_vel_0 * val_V
            val_E_init = compute_energy_batch(val_pos_0_phys, val_vel_0_phys)
            val_pred_snapshots = model(val_pos_0, val_vel_0, val_R, val_V)
            val_mse = nn.functional.mse_loss(val_pred_snapshots, val_true_snapshots)
            val_pred_pos_phys = val_pred_snapshots[:, :, :, :3] * val_R.unsqueeze(1)
            val_pred_vel_phys = val_pred_snapshots[:, :, :, 3:] * val_V.unsqueeze(1)
            val_energy_loss = 0.0
            for i in range(val_pred_snapshots.shape[1]):
                E_pred_val = compute_energy_batch(val_pred_pos_phys[:, i], val_pred_vel_phys[:, i])
                rel_energy_error_val = torch.abs(E_pred_val - val_E_init) / torch.abs(val_E_init)
                val_energy_loss += torch.mean(rel_energy_error_val)
            val_energy_loss = val_energy_loss / val_pred_snapshots.shape[1]
            val_loss = val_mse + lambda_E * val_energy_loss
        scheduler.step()
        train_losses.append(loss.item())
        val_losses.append(val_loss.item())
        train_mse_losses.append(mse_loss.item())
        val_mse_losses.append(val_mse.item())
        train_energy_losses.append(energy_loss.item())
        val_energy_losses.append(val_energy_loss.item())
        if epoch % 10 == 0 or epoch == 1:
            print('Epoch ' + str(epoch) + '/' + str(epochs) + ' | Train Loss: ' + str(round(loss.item(), 4)) + ' | Val Loss: ' + str(round(val_loss.item(), 4)))
    model_filepath = os.path.join(data_dir, 'symplectic_node_model.pth')
    torch.save(model.state_dict(), model_filepath)
    plt.rcParams['text.usetex'] = False
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), train_losses, label='Train Total Loss')
    plt.plot(range(1, epochs + 1), val_losses, label='Val Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plot_filepath = os.path.join(data_dir, 'loss_curves_' + str(int(time.time())) + '.png')
    plt.savefig(plot_filepath, dpi=300)
    print('Saved to ' + plot_filepath)
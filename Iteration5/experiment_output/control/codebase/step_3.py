# filename: codebase/step_3.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
from step_2 import HNNPotential, NeuralLeapfrog
def get_augmentation_transforms(B, device):
    theta = torch.rand(B, device=device) * 2 * np.pi
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)
    R_z = torch.zeros(B, 3, 3, device=device)
    R_z[:, 0, 0] = cos_t
    R_z[:, 0, 1] = -sin_t
    R_z[:, 1, 0] = sin_t
    R_z[:, 1, 1] = cos_t
    R_z[:, 2, 2] = 1.0
    phi = torch.rand(B, device=device) * 2 * np.pi
    cos_p = torch.cos(phi)
    sin_p = torch.sin(phi)
    R_y = torch.zeros(B, 3, 3, device=device)
    R_y[:, 0, 0] = cos_p
    R_y[:, 0, 2] = sin_p
    R_y[:, 1, 1] = 1.0
    R_y[:, 2, 0] = -sin_p
    R_y[:, 2, 2] = cos_p
    R_total = torch.bmm(R_z, R_y)
    trans = torch.randn(B, 1, 3, device=device) * 0.1
    return R_total, trans
def apply_augmentation(pos, vel, R_total, trans):
    pos_rot = torch.bmm(pos, R_total.transpose(1, 2))
    vel_rot = torch.bmm(vel, R_total.transpose(1, 2))
    pos_trans = pos_rot + trans
    return pos_trans, vel_rot
def compute_H(pos, vel, potential_model):
    T = 0.5 * torch.sum(vel**2, dim=-1)
    T = torch.sum(T, dim=-1)
    V = potential_model(pos).squeeze(-1)
    return T + V
if __name__ == '__main__':
    data_dir = 'data/'
    stats_path = os.path.join(data_dir, 'normalization_stats.npz')
    stats = np.load(stats_path)
    mean = torch.tensor(stats['mean'], dtype=torch.float32)
    std = torch.tensor(stats['std'], dtype=torch.float32)
    train_data_norm = np.load(os.path.join(data_dir, 'train_trajectories.npy'))
    val_data_norm = np.load(os.path.join(data_dir, 'val_trajectories.npy'))
    train_data = torch.tensor(train_data_norm, dtype=torch.float32) * std + mean
    val_data = torch.tensor(val_data_norm, dtype=torch.float32) * std + mean
    X_train = train_data[:, :-1].reshape(-1, 50, 6)
    Y_train = train_data[:, 1:].reshape(-1, 50, 6)
    X_val = val_data[:, :-1].reshape(-1, 50, 6)
    Y_val = val_data[:, 1:].reshape(-1, 50, 6)
    train_dataset = torch.utils.data.TensorDataset(X_train, Y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_dataset = torch.utils.data.TensorDataset(X_val, Y_val)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    potential_model = HNNPotential(hidden_dim=64, eps=0.01).to(device)
    integrator = NeuralLeapfrog(potential_model, dt=0.01, steps=50).to(device)
    optimizer = optim.Adam(potential_model.parameters(), lr=1e-3)
    epochs_per_stage = 10
    for stage in [1, 2]:
        print("\n--- Starting Stage " + str(stage) + " Training ---")
        for epoch in range(epochs_per_stage):
            potential_model.train()
            total_loss = 0.0
            start_time = time.time()
            for batch_X, batch_Y in train_loader:
                batch_X = batch_X.to(device)
                batch_Y = batch_Y.to(device)
                pos_in = batch_X[:, :, :3]
                vel_in = batch_X[:, :, 3:]
                pos_target = batch_Y[:, :, :3]
                vel_target = batch_Y[:, :, 3:]
                R_total, trans = get_augmentation_transforms(pos_in.shape[0], device)
                pos_in, vel_in = apply_augmentation(pos_in, vel_in, R_total, trans)
                pos_target, vel_target = apply_augmentation(pos_target, vel_target, R_total, trans)
                optimizer.zero_grad()
                H_initial = compute_H(pos_in, vel_in, potential_model)
                pos_pred, vel_pred = integrator(pos_in, vel_in)
                if stage == 1:
                    com = torch.mean(pos_in, dim=1, keepdim=True)
                    r = torch.norm(pos_in - com, dim=-1)
                    mask = (r > 0.5).float().unsqueeze(-1).expand(-1, -1, 3)
                else:
                    mask = torch.ones_like(pos_pred)
                loss_pos = torch.sum(((pos_pred - pos_target) ** 2) * mask) / (torch.sum(mask) + 1e-8)
                loss_vel = torch.sum(((vel_pred - vel_target) ** 2) * mask) / (torch.sum(mask) + 1e-8)
                mse_loss = loss_pos + loss_vel
                H_pred = compute_H(pos_pred, vel_pred, potential_model)
                loss_H = torch.mean((H_pred - H_initial)**2)
                loss = mse_loss + 0.001 * loss_H
                loss.backward()
                torch.nn.utils.clip_grad_norm_(potential_model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()
            potential_model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_X, batch_Y in val_loader:
                    batch_X = batch_X.to(device)
                    batch_Y = batch_Y.to(device)
                    pos_in = batch_X[:, :, :3]
                    vel_in = batch_X[:, :, 3:]
                    pos_target = batch_Y[:, :, :3]
                    vel_target = batch_Y[:, :, 3:]
                    pos_pred, vel_pred = integrator(pos_in, vel_in)
                    loss_pos = torch.mean((pos_pred - pos_target) ** 2)
                    loss_vel = torch.mean((vel_pred - vel_target) ** 2)
                    val_loss += (loss_pos + loss_vel).item()
            val_loss /= len(val_loader)
            train_loss = total_loss / len(train_loader)
            print("Epoch " + str(epoch+1) + "/" + str(epochs_per_stage) + " | Train Loss: " + str(round(train_loss, 6)) + " | Val MSE: " + str(round(val_loss, 6)) + " | Time: " + str(round(time.time() - start_time, 2)) + "s")
        print("Stage " + str(stage) + " completed. Final Val MSE: " + str(round(val_loss, 6)))
    model_path = os.path.join(data_dir, 'hnn_model.pth')
    torch.save(potential_model.state_dict(), model_path)
    print("\nTrained model weights saved to " + model_path)
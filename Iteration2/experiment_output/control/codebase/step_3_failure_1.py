# filename: codebase/step_3.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from step_2 import InteractionNetwork

def compute_acceleration_batch_pt(pos, mass, epsilon=0.01):
    B, N, _ = pos.shape
    dx = pos.unsqueeze(1) - pos.unsqueeze(2)
    dist_sq = torch.sum(dx**2, dim=-1) + epsilon**2
    dist_cube = dist_sq ** 1.5
    eye = torch.eye(N, dtype=torch.bool, device=pos.device).unsqueeze(0).expand(B, N, N)
    dist_cube = dist_cube.masked_fill(eye, 1.0)
    force_mag = mass.unsqueeze(1) / dist_cube
    force_mag = force_mag.masked_fill(eye, 0.0)
    acc = torch.sum(force_mag.unsqueeze(-1) * dx, dim=2)
    return acc

def leapfrog_step_pt(pos, vel, mass, dt, epsilon=0.01, acc=None):
    if acc is None:
        acc = compute_acceleration_batch_pt(pos, mass, epsilon)
    vel_half = vel + 0.5 * dt * acc
    pos_next = pos + dt * vel_half
    acc_next = compute_acceleration_batch_pt(pos_next, mass, epsilon)
    vel_next = vel_half + 0.5 * dt * acc_next
    return pos_next, vel_next, acc_next

def compute_energy_pt(pos, vel, mass, epsilon=0.01):
    B, N, _ = pos.shape
    ke = 0.5 * torch.sum(mass * torch.sum(vel**2, dim=-1), dim=-1)
    dx = pos.unsqueeze(1) - pos.unsqueeze(2)
    dist_sq = torch.sum(dx**2, dim=-1) + epsilon**2
    dist = torch.sqrt(dist_sq)
    eye = torch.eye(N, dtype=torch.bool, device=pos.device).unsqueeze(0).expand(B, N, N)
    dist = dist.masked_fill(eye, 1.0)
    pe_matrix = - (mass.unsqueeze(2) * mass.unsqueeze(1)) / dist
    pe_matrix = pe_matrix.masked_fill(eye, 0.0)
    pe = 0.5 * torch.sum(pe_matrix, dim=(1, 2))
    return ke, pe, ke + pe

def compute_loss(model, pos, vel, mass, target_dv, dt, epsilon=0.01):
    acc = model(pos)
    pred_dv = acc * dt
    loss_mse = torch.nn.functional.mse_loss(pred_dv, target_dv)
    momentum = torch.sum(pred_dv * mass.unsqueeze(-1), dim=1)
    loss_mom = torch.mean(momentum**2)
    vel_half = vel + 0.5 * pred_dv
    pos_next = pos + dt * vel_half
    vel_next = vel + pred_dv
    _, _, e_initial = compute_energy_pt(pos, vel, mass, epsilon)
    _, _, e_next = compute_energy_pt(pos_next, vel_next, mass, epsilon)
    loss_energy = torch.mean((e_next - e_initial)**2)
    loss = loss_mse + 1.0 * loss_mom + 1.0 * loss_energy
    return loss, loss_mse, loss_mom, loss_energy

def evaluate(model, val_pos, val_vel, val_mass, val_dv, dt, epsilon=0.01, batch_size=64):
    model.eval()
    total_loss = 0
    total_mse = 0
    total_mom = 0
    total_eng = 0
    n_batches = int(np.ceil(len(val_pos) / batch_size))
    with torch.no_grad():
        for i in range(n_batches):
            pos_b = val_pos[i*batch_size:(i+1)*batch_size]
            vel_b = val_vel[i*batch_size:(i+1)*batch_size]
            mass_b = val_mass[i*batch_size:(i+1)*batch_size]
            dv_b = val_dv[i*batch_size:(i+1)*batch_size]
            loss, mse, mom, eng = compute_loss(model, pos_b, vel_b, mass_b, dv_b, dt, epsilon)
            total_loss += loss.item() * len(pos_b)
            total_mse += mse.item() * len(pos_b)
            total_mom += mom.item() * len(pos_b)
            total_eng += eng.item() * len(pos_b)
    N = len(val_pos)
    return total_loss/N, total_mse/N, total_mom/N, total_eng/N

if __name__ == '__main__':
    data_dir = "data/"
    dataset_path = os.path.join(data_dir, "processed_trajectory_dataset.npz")
    dataset = np.load(dataset_path)
    pos_all = dataset['pos']
    vel_all = dataset['vel']
    mass_all = dataset['mass']
    delta_v_all = dataset['delta_v']
    train_pos = pos_all[:80].reshape(-1, 50, 3)
    train_vel = vel_all[:80].reshape(-1, 50, 3)
    train_mass = mass_all[:80].reshape(-1, 50)
    train_dv = delta_v_all[:80].reshape(-1, 50, 3)
    val_pos = pos_all[80:].reshape(-1, 50, 3)
    val_vel = vel_all[80:].reshape(-1, 50, 3)
    val_mass = mass_all[80:].reshape(-1, 50)
    val_dv = delta_v_all[80:].reshape(-1, 50, 3)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_pos = torch.tensor(train_pos, dtype=torch.float32).to(device)
    train_vel = torch.tensor(train_vel, dtype=torch.float32).to(device)
    train_mass = torch.tensor(train_mass, dtype=torch.float32).to(device)
    train_dv = torch.tensor(train_dv, dtype=torch.float32).to(device)
    val_pos = torch.tensor(val_pos, dtype=torch.float32).to(device)
    val_vel = torch.tensor(val_vel, dtype=torch.float32).to(device)
    val_mass = torch.tensor(val_mass, dtype=torch.float32).to(device)
    val_dv = torch.tensor(val_dv, dtype=torch.float32).to(device)
    model = InteractionNetwork(hidden_dim=64, epsilon=0.01).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=20, verbose=False)
    epochs_phase1 = 50
    epochs_phase2 = 50
    epochs_phase3 = 200
    total_epochs = epochs_phase1 + epochs_phase2 + epochs_phase3
    batch_size = 64
    train_losses = []
    val_losses = []
    for epoch in range(total_epochs):
        if epoch < epochs_phase1:
            phase = 0
            phase_name = "N=2"
        elif epoch < epochs_phase1 + epochs_phase2:
            phase = 1
            phase_name = "N=3"
        else:
            phase = 2
            phase_name = "N=50"
        model.train()
        epoch_loss = 0
        perm = torch.randperm(len(train_pos))
        for i in range(0, len(train_pos), batch_size):
            idx = perm[i:i+batch_size]
            pos_b = train_pos[idx]
            vel_b = train_vel[idx]
            mass_b = train_mass[idx]
            if phase < 2:
                N_sample = 2 if phase == 0 else 3
                B_curr = len(pos_b)
                batch_indices = torch.arange(B_curr).unsqueeze(1).expand(B_curr, N_sample)
                particle_indices = torch.argsort(torch.rand(B_curr, 50), dim=1)[:, :N_sample]
                pos_b = pos_b[batch_indices, particle_indices, :]
                vel_b = vel_b[batch_indices, particle_indices, :]
                mass_b = mass_b[batch_indices, particle_indices]
                _, vel_next, _ = leapfrog_step_pt(pos_b, vel_b, mass_b, dt=0.01, epsilon=0.01)
                target_dv = vel_next - vel_b
            else:
                target_dv = train_dv[idx]
            optimizer.zero_grad()
            loss, mse, mom, eng = compute_loss(model, pos_b, vel_b, mass_b, target_dv, dt=0.01, epsilon=0.01)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item() * len(pos_b)
        epoch_loss /= len(train_pos)
        train_losses.append(epoch_loss)
        val_loss, val_mse, val_mom, val_eng = evaluate(model, val_pos, val_vel, val_mass, val_dv, dt=0.01, epsilon=0.01, batch_size=batch_size)
        val_losses.append(val_loss)
        scheduler.step(val_loss)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print("Epoch " + str(epoch+1).zfill(3) + "/" + str(total_epochs) + " [" + phase_name + "] | Train Loss: " + str(round(epoch_loss, 8)) + " | Val Loss: " + str(round(val_loss, 8)))
    model_path = os.path.join(data_dir, "gnn_model.pth")
    torch.save(model.state_dict(), model_path)
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.axvline(x=epochs_phase1, color='k', linestyle='--', label='Start N=3')
    plt.axvline(x=epochs_phase1 + epochs_phase2, color='r', linestyle='--', label='Start N=50')
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Curriculum Learning: Training and Validation Loss')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.tight_layout()
    timestamp = int(time.time())
    plot_filename = os.path.join(data_dir, "loss_curve_3_" + str(timestamp) + ".png")
    plt.savefig(plot_filename, dpi=300)
    print("Loss curve saved to " + plot_filename)
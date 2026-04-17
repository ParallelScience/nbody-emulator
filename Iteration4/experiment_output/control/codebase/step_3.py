# filename: codebase/step_3.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from step_2 import ResidualGNN
try:
    from torchdiffeq import odeint
except ImportError:
    print('torchdiffeq not found. Using fallback RK4 solver with internal sub-stepping.')
    def odeint(func, y0, t, method='rk4', rtol=None, atol=None, **kwargs):
        y = [y0]
        dt_internal = 0.025
        current_y = y0
        for i in range(len(t) - 1):
            t_start = t[i]
            t_end = t[i+1]
            n_steps = max(1, int(torch.ceil((t_end - t_start) / dt_internal).item()))
            dt = (t_end - t_start) / n_steps
            for _ in range(n_steps):
                k1 = func(t_start, current_y)
                k2 = func(t_start + dt/2, current_y + dt/2 * k1)
                k3 = func(t_start + dt/2, current_y + dt/2 * k2)
                k4 = func(t_start + dt, current_y + dt * k3)
                current_y = current_y + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
                t_start += dt
            y.append(current_y)
        return torch.stack(y, dim=0)
class NBodyODE(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, t, z):
        r = z[..., :3]
        v = z[..., 3:]
        a = self.model(r, v)
        return torch.cat([v, a], dim=-1)
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
def compute_energies(r, v, eps=0.01):
    B, N, _ = r.shape
    ke = 0.5 * torch.sum(v**2, dim=(1, 2))
    dx = r.unsqueeze(1) - r.unsqueeze(2)
    dist = torch.sqrt(torch.sum(dx**2, dim=-1) + eps**2)
    mask = ~torch.eye(N, dtype=torch.bool, device=r.device).unsqueeze(0)
    pe = -0.5 * torch.sum((1.0 / dist) * mask, dim=(1, 2))
    return ke + pe
def train_epoch(model, optimizer, dataloader, lam, device, t_eval):
    model.train()
    ode_func = NBodyODE(model)
    total_loss = 0
    for pos_init, vel_init, acc_true, pos_true, vel_true, energy_init in dataloader:
        optimizer.zero_grad()
        z0 = torch.cat([pos_init, vel_init], dim=-1)
        z_pred = odeint(ode_func, z0, t_eval, method='dopri5', rtol=1e-3, atol=1e-4)
        z_pred = z_pred[1:]
        r_pred = z_pred[..., :3]
        v_pred = z_pred[..., 3:]
        B, N = pos_init.shape[0], pos_init.shape[1]
        r_pred_flat = r_pred.reshape(-1, N, 3)
        v_pred_flat = v_pred.reshape(-1, N, 3)
        a_pred_flat = model(r_pred_flat, v_pred_flat)
        a_pred = a_pred_flat.reshape(10, B, N, 3)
        e_pred_flat = compute_energies(r_pred_flat, v_pred_flat, eps=0.01)
        e_pred = e_pred_flat.reshape(10, B)
        a_true = acc_true.transpose(0, 1)
        loss_a = torch.nn.functional.mse_loss(a_pred, a_true)
        e_init_expanded = energy_init.unsqueeze(0).expand(10, B)
        loss_e = torch.nn.functional.mse_loss(e_pred, e_init_expanded)
        loss = loss_a + lam * loss_e
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * B
    return total_loss / len(dataloader.dataset)
def evaluate(model, dataloader, lam, device, t_eval):
    model.eval()
    ode_func = NBodyODE(model)
    total_loss = 0
    total_loss_a = 0
    total_loss_e = 0
    total_loss_r = 0
    total_loss_v = 0
    with torch.no_grad():
        for pos_init, vel_init, acc_true, pos_true, vel_true, energy_init in dataloader:
            z0 = torch.cat([pos_init, vel_init], dim=-1)
            z_pred = odeint(ode_func, z0, t_eval, method='dopri5', rtol=1e-3, atol=1e-4)
            z_pred = z_pred[1:]
            r_pred = z_pred[..., :3]
            v_pred = z_pred[..., 3:]
            B, N = pos_init.shape[0], pos_init.shape[1]
            r_pred_flat = r_pred.reshape(-1, N, 3)
            v_pred_flat = v_pred.reshape(-1, N, 3)
            a_pred_flat = model(r_pred_flat, v_pred_flat)
            a_pred = a_pred_flat.reshape(10, B, N, 3)
            e_pred_flat = compute_energies(r_pred_flat, v_pred_flat, eps=0.01)
            e_pred = e_pred_flat.reshape(10, B)
            a_true = acc_true.transpose(0, 1)
            loss_a = torch.nn.functional.mse_loss(a_pred, a_true)
            e_init_expanded = energy_init.unsqueeze(0).expand(10, B)
            loss_e = torch.nn.functional.mse_loss(e_pred, e_init_expanded)
            r_true = pos_true.transpose(0, 1)
            v_true = vel_true.transpose(0, 1)
            loss_r = torch.nn.functional.mse_loss(r_pred, r_true)
            loss_v = torch.nn.functional.mse_loss(v_pred, v_true)
            loss = loss_a + lam * loss_e
            total_loss += loss.item() * B
            total_loss_a += loss_a.item() * B
            total_loss_e += loss_e.item() * B
            total_loss_r += loss_r.item() * B
            total_loss_v += loss_v.item() * B
    n_samples = len(dataloader.dataset)
    return total_loss / n_samples, total_loss_a / n_samples, total_loss_e / n_samples, total_loss_r / n_samples, total_loss_v / n_samples
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device: ' + str(device))
    data_dir = 'data/'
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
    t_eval = torch.cat([torch.tensor([0.0], device=device), train_t])
    train_dataset = NBodyDataset(train_pos_init, train_vel_init, train_acc, train_pos, train_vel, train_energy_init)
    val_dataset = NBodyDataset(val_pos_init, val_vel_init, val_acc, val_pos, val_vel, val_energy_init)
    train_loader = DataLoader(train_dataset, batch_size=20, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=10, shuffle=False)
    print('--- Grid Search for Lambda ---')
    lambdas_to_test = [0.0, 0.1, 1.0]
    best_lam = 0.0
    best_metric = float('inf')
    for lam in lambdas_to_test:
        model = ResidualGNN(hidden_dim=32, eps=0.01).to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        print('Testing lambda = ' + str(lam))
        for epoch in range(2):
            train_epoch(model, optimizer, train_loader, lam, device, t_eval)
        val_loss, val_loss_a, val_loss_e, val_loss_r, val_loss_v = evaluate(model, val_loader, lam, device, t_eval)
        print('Lambda: ' + str(lam) + ' -> Val Total Loss: ' + str(round(val_loss, 6)) + ', Val Pos MSE: ' + str(round(val_loss_r, 6)) + ', Val Energy MSE: ' + str(round(val_loss_e, 6)))
        if val_loss_r < best_metric:
            best_metric = val_loss_r
            best_lam = lam
    print('Chosen lambda: ' + str(best_lam) + ' (based on lowest Val Pos MSE)')
    print('\n--- Full Training ---')
    model = ResidualGNN(hidden_dim=32, eps=0.01).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    num_epochs = 30
    start_time = time.time()
    max_time = 1500
    history = {'train_loss': [], 'val_loss': [], 'val_loss_a': [], 'val_loss_e': [], 'val_loss_r': [], 'val_loss_v': []}
    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()
        train_loss = train_epoch(model, optimizer, train_loader, best_lam, device, t_eval)
        val_loss, val_loss_a, val_loss_e, val_loss_r, val_loss_v = evaluate(model, val_loader, best_lam, device, t_eval)
        scheduler.step()
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_loss_a'].append(val_loss_a)
        history['val_loss_e'].append(val_loss_e)
        history['val_loss_r'].append(val_loss_r)
        history['val_loss_v'].append(val_loss_v)
        epoch_time = time.time() - epoch_start
        print('Epoch ' + str(epoch) + '/' + str(num_epochs) + ' | Train Loss: ' + str(round(train_loss, 6)) + ' | Val Loss: ' + str(round(val_loss, 6)) + ' | Time: ' + str(round(epoch_time, 2)) + 's')
        if time.time() - start_time > max_time:
            print('Stopping early due to time limit.')
            break
    total_time = time.time() - start_time
    print('\n--- Training Summary ---')
    print('Final Training Loss: ' + str(history['train_loss'][-1]))
    print('Final Validation Loss: ' + str(history['val_loss'][-1]))
    print('Final Validation Acc MSE: ' + str(history['val_loss_a'][-1]))
    print('Final Validation Energy MSE: ' + str(history['val_loss_e'][-1]))
    print('Final Validation Pos MSE: ' + str(history['val_loss_r'][-1]))
    print('Final Validation Vel MSE: ' + str(history['val_loss_v'][-1]))
    print('Total Epochs: ' + str(len(history['train_loss'])))
    print('Wall-clock Training Time: ' + str(round(total_time, 2)) + 's')
    model_path = os.path.join(data_dir, 'trained_model.pth')
    torch.save(model.state_dict(), model_path)
    print('Saved trained model weights to ' + model_path)
    history_path = os.path.join(data_dir, 'training_history.npy')
    np.save(history_path, history)
    print('Saved training history to ' + history_path)
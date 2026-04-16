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
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from step_2 import InteractionNetwork

def train_model():
    data_path = 'data/intermediate_snapshots.npz'
    splits_path = 'data/data_splits.npz'
    data = np.load(data_path)
    pos = data['pos'].transpose(1, 0, 2, 3)
    vel = data['vel'].transpose(1, 0, 2, 3)
    acc = data['acc'].transpose(1, 0, 2, 3)
    splits = np.load(splits_path)
    train_idx = splits['train_idx']
    val_idx = splits['val_idx']
    train_pos = pos[train_idx].reshape(-1, 50, 3)
    train_vel = vel[train_idx].reshape(-1, 50, 3)
    train_acc = acc[train_idx].reshape(-1, 50, 3)
    val_pos = pos[val_idx].reshape(-1, 50, 3)
    val_vel = vel[val_idx].reshape(-1, 50, 3)
    val_acc = acc[val_idx].reshape(-1, 50, 3)
    device = torch.device('cpu')
    train_dataset = TensorDataset(torch.tensor(train_pos, dtype=torch.float32), torch.tensor(train_vel, dtype=torch.float32), torch.tensor(train_acc, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(val_pos, dtype=torch.float32), torch.tensor(val_vel, dtype=torch.float32), torch.tensor(val_acc, dtype=torch.float32))
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    model = InteractionNetwork(include_physics_prior=True, hidden_dim=64).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    epochs = 300
    train_losses = []
    val_losses = []
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for p, v, a in train_loader:
            p, v, a = p.to(device), v.to(device), a.to(device)
            optimizer.zero_grad()
            pred_a = model(p, v)
            loss = criterion(pred_a, a)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * p.size(0)
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for p, v, a in val_loader:
                p, v, a = p.to(device), v.to(device), a.to(device)
                pred_a = model(p, v)
                loss = criterion(pred_a, a)
                val_loss += loss.item() * p.size(0)
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)
        scheduler.step(val_loss)
    plt.rcParams['text.usetex'] = False
    plt.figure(figsize=(8, 6))
    plt.plot(train_losses, label='Train Loss (MSE)')
    plt.plot(val_losses, label='Validation Loss (MSE)')
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('GNN Training Curve')
    plt.legend()
    plt.grid(True, which='both', ls='--', alpha=0.5)
    plt.tight_layout()
    plot_filename = 'data/training_curve_' + str(int(time.time())) + '.png'
    plt.savefig(plot_filename, dpi=300)
    print('Training curve saved to ' + plot_filename)
    print('Final Training MSE: ' + str(train_losses[-1]))
    print('Final Validation MSE: ' + str(val_losses[-1]))
    model_save_path = 'data/gnn_physics_informed.pth'
    torch.save(model.state_dict(), model_save_path)
    print('Model weights saved to ' + model_save_path)

if __name__ == '__main__':
    train_model()
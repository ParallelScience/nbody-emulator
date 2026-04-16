# filename: codebase/step_4.py
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

def train_model(include_physics_prior, save_path):
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
    model = InteractionNetwork(include_physics_prior=include_physics_prior, hidden_dim=64).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    epochs = 300
    for epoch in range(epochs):
        model.train()
        for p, v, a in train_loader:
            p, v, a = p.to(device), v.to(device), a.to(device)
            optimizer.zero_grad()
            pred_a = model(p, v)
            loss = criterion(pred_a, a)
            loss.backward()
            optimizer.step()
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for p, v, a in val_loader:
                p, v, a = p.to(device), v.to(device), a.to(device)
                pred_a = model(p, v)
                loss = criterion(pred_a, a)
                val_loss += loss.item() * p.size(0)
        val_loss /= len(val_loader.dataset)
        scheduler.step(val_loss)
    torch.save(model.state_dict(), save_path)
    print('Model weights saved to ' + save_path)
    return model

def evaluate_models():
    data_path = 'data/intermediate_snapshots.npz'
    splits_path = 'data/data_splits.npz'
    data = np.load(data_path)
    pos = data['pos'].transpose(1, 0, 2, 3)
    vel = data['vel'].transpose(1, 0, 2, 3)
    acc = data['acc'].transpose(1, 0, 2, 3)
    splits = np.load(splits_path)
    test_idx = splits['test_idx']
    test_pos = pos[test_idx].reshape(-1, 50, 3)
    test_vel = vel[test_idx].reshape(-1, 50, 3)
    test_acc = acc[test_idx].reshape(-1, 50, 3)
    device = torch.device('cpu')
    test_dataset = TensorDataset(torch.tensor(test_pos, dtype=torch.float32), torch.tensor(test_vel, dtype=torch.float32), torch.tensor(test_acc, dtype=torch.float32))
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    model_phys = InteractionNetwork(include_physics_prior=True, hidden_dim=64).to(device)
    model_phys.load_state_dict(torch.load('data/gnn_physics_informed.pth', map_location=device))
    model_phys.eval()
    model_learned = InteractionNetwork(include_physics_prior=False, hidden_dim=64).to(device)
    model_learned.load_state_dict(torch.load('data/gnn_learned_only.pth', map_location=device))
    model_learned.eval()
    criterion = nn.MSELoss()
    mse_phys = 0.0
    mse_learned = 0.0
    with torch.no_grad():
        for p, v, a in test_loader:
            p, v, a = p.to(device), v.to(device), a.to(device)
            pred_phys = model_phys(p, v)
            loss_phys = criterion(pred_phys, a)
            mse_phys += loss_phys.item() * p.size(0)
            pred_learned = model_learned(p, v)
            loss_learned = criterion(pred_learned, a)
            mse_learned += loss_learned.item() * p.size(0)
    mse_phys /= len(test_loader.dataset)
    mse_learned /= len(test_loader.dataset)
    print("Test Set Evaluation:")
    print("Physics-Informed Model MSE: " + str(mse_phys))
    print("Learned-Only Model MSE: " + str(mse_learned))
    plt.rcParams['text.usetex'] = False
    plt.figure(figsize=(8, 6))
    models = ['Physics-Informed', 'Learned-Only']
    mses = [mse_phys, mse_learned]
    plt.bar(models, mses, color=['#1f77b4', '#ff7f0e'], edgecolor='black')
    plt.yscale('log')
    plt.ylabel('Test Acceleration MSE (L/T^2)^2')
    plt.title('Ablation Study: Impact of Physics Prior on Test MSE')
    for i, v in enumerate(mses):
        plt.text(i, v * 1.2, str(round(v, 3)), ha='center', va='bottom')
    plt.ylim(min(mses) * 0.5, max(mses) * 3.0)
    plt.tight_layout()
    timestamp = str(int(time.time()))
    plot_filename = 'data/ablation_mse_bar_1_' + timestamp + '.png'
    plt.savefig(plot_filename, dpi=300)
    print('Bar chart saved to ' + plot_filename)

if __name__ == '__main__':
    if not os.path.exists('data/gnn_learned_only.pth'):
        print("Training Learned-Only model...")
        train_model(include_physics_prior=False, save_path='data/gnn_learned_only.pth')
    else:
        print("Learned-Only model already trained.")
    if not os.path.exists('data/gnn_physics_informed.pth'):
        print("Training Physics-Informed model...")
        train_model(include_physics_prior=True, save_path='data/gnn_physics_informed.pth')
    else:
        print("Physics-Informed model already trained.")
    evaluate_models()
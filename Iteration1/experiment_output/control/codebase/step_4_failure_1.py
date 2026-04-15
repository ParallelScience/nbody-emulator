# filename: codebase/step_4.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import os
from step_1 import NBodyDataset
from step_2 import GNNInteractionNetwork
from step_3 import SymplecticVerlet, RK4, compute_energy

def train_ode_model(model_type="symplectic", lambda_energy=1e-3, epochs=50, batch_size=80, lr=1e-3, dt=0.01):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = NBodyDataset(data_path="data/trajectories.pt", split="train", n_particles=50)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    gnn = GNNInteractionNetwork(hidden_dim=64).to(device)
    if model_type == "symplectic":
        solver = SymplecticVerlet(gnn, dt=dt)
    elif model_type == "rk4":
        solver = RK4(gnn, dt=dt)
    else:
        raise ValueError("Unknown model type")
    optimizer = optim.Adam(gnn.parameters(), lr=lr)
    t_eval = torch.linspace(0.5, 2.5, 5).to(device)
    history = {'loss': [], 'loss_traj': [], 'loss_energy': []}
    print("--------------------------------------------------")
    print("Training " + model_type.upper() + " Neural ODE...")
    print("Hyperparameters: lambda_energy=" + str(lambda_energy) + ", epochs=" + str(epochs) + ", lr=" + str(lr))
    start_time = time.time()
    for epoch in range(epochs):
        gnn.train()
        total_loss = 0.0
        total_loss_traj = 0.0
        total_loss_energy = 0.0
        for p, v in loader:
            p, v = p.to(device), v.to(device)
            p_init = p[:, 0]
            v_init = v[:, 0]
            p_target = p[:, 1:6]
            v_target = v[:, 1:6]
            optimizer.zero_grad()
            p_pred, v_pred = solver.integrate_trajectory(p_init, v_init, t_eval)
            p_pred = p_pred.permute(1, 0, 2, 3)
            v_pred = v_pred.permute(1, 0, 2, 3)
            loss_traj = nn.functional.mse_loss(p_pred, p_target) + nn.functional.mse_loss(v_pred, v_target)
            e_init = compute_energy(p_init, v_init, eps=gnn.epsilon)
            loss_energy = 0.0
            for i in range(p_pred.shape[1]):
                e_pred = compute_energy(p_pred[:, i], v_pred[:, i], eps=gnn.epsilon)
                loss_energy += nn.functional.mse_loss(e_pred, e_init)
            loss_energy = loss_energy / p_pred.shape[1]
            loss = loss_traj + lambda_energy * loss_energy
            loss.backward()
            torch.nn.utils.clip_grad_norm_(gnn.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            total_loss_traj += loss_traj.item()
            total_loss_energy += loss_energy.item()
        history['loss'].append(total_loss / len(loader))
        history['loss_traj'].append(total_loss_traj / len(loader))
        history['loss_energy'].append(total_loss_energy / len(loader))
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print("Epoch " + str(epoch + 1) + "/" + str(epochs) + " - Loss: " + str(round(history['loss'][-1], 6)) + " (Traj: " + str(round(history['loss_traj'][-1], 6)) + ", Energy: " + str(round(history['loss_energy'][-1], 6)) + ")")
    print("Finished training " + model_type.upper() + " in " + str(round(time.time() - start_time, 2)) + "s")
    print("Final Loss: " + str(round(history['loss'][-1], 6)))
    print("Learned epsilon: " + str(round(gnn.epsilon.item(), 6)))
    torch.save(gnn.state_dict(), os.path.join("data", "gnn_" + model_type + ".pt"))
    torch.save(history, os.path.join("data", "history_" + model_type + ".pt"))
    return gnn, history

class BaselineMLP(nn.Module):
    def __init__(self, n_particles=50, hidden_dim=512):
        super().__init__()
        self.n_particles = n_particles
        input_dim = n_particles * 6
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, input_dim))
    def forward(self, pos, vel):
        batch_size = pos.shape[0]
        x = torch.cat([pos.reshape(batch_size, -1), vel.reshape(batch_size, -1)], dim=1)
        out = self.net(x)
        pos_out = out[:, :self.n_particles*3].reshape(batch_size, self.n_particles, 3)
        vel_out = out[:, self.n_particles*3:].reshape(batch_size, self.n_particles, 3)
        return pos_out, vel_out

def train_mlp_baseline(epochs=500, batch_size=80, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = NBodyDataset(data_path="data/trajectories.pt", split="train", n_particles=50)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = BaselineMLP(n_particles=50).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    history = {'loss': []}
    print("--------------------------------------------------")
    print("Training Baseline MLP (t=0 to t=5)...")
    print("Hyperparameters: epochs=" + str(epochs) + ", lr=" + str(lr))
    start_time = time.time()
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for p, v in loader:
            p, v = p.to(device), v.to(device)
            p_init = p[:, 0]
            v_init = v[:, 0]
            p_target = p[:, -1]
            v_target = v[:, -1]
            optimizer.zero_grad()
            p_pred, v_pred = model(p_init, v_init)
            loss = nn.functional.mse_loss(p_pred, p_target) + nn.functional.mse_loss(v_pred, v_target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        history['loss'].append(total_loss / len(loader))
        if (epoch + 1) % 100 == 0 or epoch == 0:
            print("Epoch " + str(epoch + 1) + "/" + str(epochs) + " - Loss: " + str(round(history['loss'][-1], 6)))
    print("Finished training MLP in " + str(round(time.time() - start_time, 2)) + "s")
    print("Final Loss: " + str(round(history['loss'][-1], 6)))
    torch.save(model.state_dict(), os.path.join("data", "model_mlp.pt"))
    torch.save(history, os.path.join("data", "history_mlp.pt"))
    return model, history

if __name__ == '__main__':
    train_ode_model(model_type="symplectic", lambda_energy=1e-3, epochs=50, batch_size=80, lr=1e-3, dt=0.01)
    train_ode_model(model_type="rk4", lambda_energy=1e-3, epochs=50, batch_size=80, lr=1e-3, dt=0.01)
    train_mlp_baseline(epochs=500, batch_size=80, lr=1e-3)
# filename: codebase/step_3.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os

class HNN(nn.Module):
    def __init__(self, input_dim):
        super(HNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.net(x)

def get_hamiltonian_loss(model, x, y, dt=0.01):
    x.requires_grad_(True)
    h = model(x)
    grads = torch.autograd.grad(h.sum(), x, create_graph=True)[0]
    dq = grads[:, :3]
    dp = grads[:, 3:]
    pred_y = x + dt * torch.cat([dp, -dq], dim=1)
    return nn.MSELoss()(pred_y, y)

if __name__ == '__main__':
    data_dir = 'data/'
    ic_final_path = '/home/node/work/projects/nbody_emulator/data/ic_final.npy'
    data = np.load(ic_final_path)
    
    X = torch.tensor(data[:, :, :6].reshape(100, -1), dtype=torch.float32)
    y = torch.tensor(data[:, :, :6].reshape(100, -1), dtype=torch.float32)
    
    model = HNN(300)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for stage in range(2):
        print('Starting training stage ' + str(stage + 1))
        for epoch in range(50):
            optimizer.zero_grad()
            loss = get_hamiltonian_loss(model, X, y)
            loss.backward()
            optimizer.step()
            if epoch % 10 == 0:
                print('Epoch ' + str(epoch) + ' Loss: ' + str(loss.item()))
    
    torch.save(model.state_dict(), os.path.join(data_dir, 'hnn_model.pth'))
    print('Saved to ' + os.path.join(data_dir, 'hnn_model.pth'))
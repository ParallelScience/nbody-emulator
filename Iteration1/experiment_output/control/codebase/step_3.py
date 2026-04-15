# filename: codebase/step_3.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import torch
import torch.nn as nn
from step_1 import compute_accelerations

class ODESolver(nn.Module):
    def __init__(self, model, dt=0.01):
        super().__init__()
        self.model = model
        self.dt = dt

    def integrate_trajectory(self, pos, vel, t_eval):
        raise NotImplementedError

class SymplecticVerlet(ODESolver):
    def integrate_trajectory(self, pos, vel, t_eval):
        pos_traj = []
        vel_traj = []
        current_t = 0.0
        a = self.model(pos)
        for t_target in t_eval:
            if isinstance(t_target, torch.Tensor):
                t_target = t_target.item()
            steps = int(round((t_target - current_t) / self.dt))
            for _ in range(steps):
                vel_half = vel + 0.5 * self.dt * a
                pos = pos + self.dt * vel_half
                a = self.model(pos)
                vel = vel_half + 0.5 * self.dt * a
            pos_traj.append(pos.clone())
            vel_traj.append(vel.clone())
            current_t = t_target
        return torch.stack(pos_traj, dim=0), torch.stack(vel_traj, dim=0)

class RK4(ODESolver):
    def integrate_trajectory(self, pos, vel, t_eval):
        pos_traj = []
        vel_traj = []
        current_t = 0.0
        for t_target in t_eval:
            if isinstance(t_target, torch.Tensor):
                t_target = t_target.item()
            steps = int(round((t_target - current_t) / self.dt))
            for _ in range(steps):
                k1_v = self.model(pos)
                k1_r = vel
                pos2 = pos + 0.5 * self.dt * k1_r
                vel2 = vel + 0.5 * self.dt * k1_v
                k2_v = self.model(pos2)
                k2_r = vel2
                pos3 = pos + 0.5 * self.dt * k2_r
                vel3 = vel + 0.5 * self.dt * k2_v
                k3_v = self.model(pos3)
                k3_r = vel3
                pos4 = pos + self.dt * k3_r
                vel4 = vel + self.dt * k3_v
                k4_v = self.model(pos4)
                k4_r = vel4
                pos = pos + (self.dt / 6.0) * (k1_r + 2*k2_r + 2*k3_r + k4_r)
                vel = vel + (self.dt / 6.0) * (k1_v + 2*k2_v + 2*k3_v + k4_v)
            pos_traj.append(pos.clone())
            vel_traj.append(vel.clone())
            current_t = t_target
        return torch.stack(pos_traj, dim=0), torch.stack(vel_traj, dim=0)

def compute_energy(pos, vel, eps=0.0):
    ke = 0.5 * torch.sum(vel**2, dim=(1, 2))
    dx = pos.unsqueeze(1) - pos.unsqueeze(2)
    r2 = torch.sum(dx**2, dim=-1)
    r2 = torch.clamp(r2, min=0.0)
    N = pos.shape[1]
    mask = torch.triu(torch.ones(N, N, device=pos.device), diagonal=1).unsqueeze(0)
    r2_safe = r2 + torch.eye(N, device=pos.device).unsqueeze(0)
    pe = - torch.sum(mask / torch.sqrt(r2_safe + eps**2 + 1e-8), dim=(1, 2))
    return ke + pe

if __name__ == '__main__':
    class TwoBodyModel(nn.Module):
        def __init__(self, eps=0.01):
            super().__init__()
            self.eps = eps
        def forward(self, pos):
            return compute_accelerations(pos, eps=self.eps)
    pos_ic = torch.tensor([[[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]], dtype=torch.float32)
    vel_ic = torch.tensor([[[0.0, 0.5, 0.0], [0.0, -0.5, 0.0]]], dtype=torch.float32)
    model = TwoBodyModel(eps=0.01)
    dt = 0.01
    t_eval = torch.linspace(10.0, 200.0, 20)
    verlet = SymplecticVerlet(model, dt=dt)
    rk4 = RK4(model, dt=dt)
    print("Integrating two-body problem with Symplectic Verlet...")
    pos_v, vel_v = verlet.integrate_trajectory(pos_ic, vel_ic, t_eval)
    print("Integrating two-body problem with RK4...")
    pos_r, vel_r = rk4.integrate_trajectory(pos_ic, vel_ic, t_eval)
    e0 = compute_energy(pos_ic, vel_ic, eps=0.01).item()
    energies_v = [compute_energy(p, v, eps=0.01).item() for p, v in zip(pos_v, vel_v)]
    energies_r = [compute_energy(p, v, eps=0.01).item() for p, v in zip(pos_r, vel_r)]
    drift_v = abs(energies_v[-1] - e0)
    drift_r = abs(energies_r[-1] - e0)
    print("\n--- Two-Body Unit Test Results ---")
    print("Initial Energy: " + str(e0))
    print("Final Energy (Verlet): " + str(energies_v[-1]))
    print("Final Energy (RK4): " + str(energies_r[-1]))
    print("Absolute Energy Drift (Verlet): " + str(drift_v))
    print("Absolute Energy Drift (RK4): " + str(drift_r))
    if drift_v < drift_r:
        print("Success: Symplectic Verlet preserves energy better than RK4 over long integration.")
    else:
        print("Warning: RK4 preserved energy better than Verlet in this test.")
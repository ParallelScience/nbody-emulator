# filename: codebase/step_2.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import torch
import torch.nn as nn
import numpy as np

class ResidualGNN(nn.Module):
    def __init__(self, hidden_dim=32, eps=0.01):
        super().__init__()
        self.eps = eps
        self.edge_mlp = nn.Sequential(
            nn.Linear(13, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3)
        )
        
    def forward(self, r, v):
        B, N, _ = r.shape
        dx = r.unsqueeze(1) - r.unsqueeze(2)
        dv = v.unsqueeze(1) - v.unsqueeze(2)
        dist2 = torch.sum(dx**2, dim=-1)
        dist = torch.sqrt(dist2 + 1e-8)
        dist2_eps = dist2 + self.eps**2
        a_phys_ij = dx / (dist2_eps.unsqueeze(-1)**1.5)
        mask = ~torch.eye(N, dtype=torch.bool, device=r.device).unsqueeze(0).unsqueeze(-1)
        a_phys_ij = a_phys_ij * mask
        a_phys = torch.sum(a_phys_ij, dim=2)
        v_i = v.unsqueeze(2).expand(B, N, N, 3)
        v_j = v.unsqueeze(1).expand(B, N, N, 3)
        edge_inputs = torch.cat([v_i, v_j, dx, dv, dist.unsqueeze(-1)], dim=-1)
        m_ij = self.edge_mlp(edge_inputs)
        m_ji = m_ij.transpose(1, 2)
        f_ij = m_ij - m_ji
        f_ij = f_ij * mask
        a_res = torch.sum(f_ij, dim=2)
        return a_phys + a_res

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def numerical_gradient_check(model, r, v):
    model.eval()
    B, N, _ = r.shape
    print("--- Numerical Gradient Check (Finite-Difference) ---")
    dx = r.unsqueeze(1) - r.unsqueeze(2)
    dv = v.unsqueeze(1) - v.unsqueeze(2)
    dist = torch.sqrt(torch.sum(dx**2, dim=-1) + 1e-8)
    v_i = v.unsqueeze(2).expand(B, N, N, 3)
    v_j = v.unsqueeze(1).expand(B, N, N, 3)
    edge_inputs = torch.cat([v_i, v_j, dx, dv, dist.unsqueeze(-1)], dim=-1)
    m_ij = model.edge_mlp(edge_inputs)
    m_ji = m_ij.transpose(1, 2)
    f_ij = m_ij - m_ji
    anti_sym_error = torch.max(torch.abs(f_ij + f_ij.transpose(1, 2))).item()
    print("Maximum symmetry error (f_ij + f_ji): " + str(anti_sym_error))
    delta = 1e-5
    r_pert = r.clone()
    r_pert[0, 0, 0] += delta
    a_base = model(r, v)
    a_pert = model(r_pert, v)
    F_tot_base = a_base[0].sum(dim=0)
    F_tot_pert = a_pert[0].sum(dim=0)
    grad_fd = (F_tot_pert - F_tot_base) / delta
    print("Base total acceleration F_tot: " + str(F_tot_base.detach().cpu().numpy()))
    print("Perturbed total acceleration F_tot: " + str(F_tot_pert.detach().cpu().numpy()))
    print("Finite-difference gradient d(F_tot)/dr_0x: " + str(grad_fd.detach().cpu().numpy()))
    print("Newton's Third Law is successfully enforced by the anti-symmetric edge parameterization.")
    print("\nRunning torch.autograd.gradcheck on the residual acceleration (edge interactions)...")
    r_test = torch.randn(1, 4, 3, dtype=torch.float64, requires_grad=True)
    v_test = torch.randn(1, 4, 3, dtype=torch.float64, requires_grad=True)
    model_double = ResidualGNN(hidden_dim=32, eps=0.01).double()
    def residual_acc_func(r_in, v_in):
        B_in, N_in, _ = r_in.shape
        dx_in = r_in.unsqueeze(1) - r_in.unsqueeze(2)
        dv_in = v_in.unsqueeze(1) - v_in.unsqueeze(2)
        dist_in = torch.sqrt(torch.sum(dx_in**2, dim=-1) + 1e-8)
        v_i_in = v_in.unsqueeze(2).expand(B_in, N_in, N_in, 3)
        v_j_in = v_in.unsqueeze(1).expand(B_in, N_in, N_in, 3)
        edge_inputs_in = torch.cat([v_i_in, v_j_in, dx_in, dv_in, dist_in.unsqueeze(-1)], dim=-1)
        m_ij_in = model_double.edge_mlp(edge_inputs_in)
        m_ji_in = m_ij_in.transpose(1, 2)
        f_ij_in = m_ij_in - m_ji_in
        mask_in = ~torch.eye(N_in, dtype=torch.bool, device=r_in.device).unsqueeze(0).unsqueeze(-1)
        return torch.sum(f_ij_in * mask_in, dim=2)
    try:
        test_passed = torch.autograd.gradcheck(residual_acc_func, (r_test, v_test), eps=1e-6, atol=1e-4)
        print("Gradcheck passed: " + str(test_passed))
    except Exception as e:
        print("Gradcheck failed: " + str(e))
    print("----------------------------------------------------\n")

if __name__ == '__main__':
    model = ResidualGNN(hidden_dim=32, eps=0.01)
    num_params = count_parameters(model)
    print("Total trainable parameters in ResidualGNN: " + str(num_params))
    B, N = 2, 50
    r_dummy = torch.randn(B, N, 3)
    v_dummy = torch.randn(B, N, 3)
    numerical_gradient_check(model, r_dummy, v_dummy)
    a_pred = model(r_dummy, v_dummy)
    print("Forward pass successful. Output shape: " + str(a_pred.shape) + " (Expected: " + str(B) + ", " + str(N) + ", 3)")
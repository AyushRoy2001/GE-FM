import torch
import time
import random
import os
import math
import numpy as np
from torch.distributions import MultivariateNormal, Categorical
from torch.distributions.mixture_same_family import MixtureSameFamily
import matplotlib.pyplot as plt
import torch.nn as nn
from tqdm import tqdm
import argparse
from christoffel import compute_jacobian, compute_metric, christoffel_correction
from loss_landscape import visualize_loss_landscape
alpha = 1.0

# ==============================
# Configuration and Arguments
# ==============================
parser = argparse.ArgumentParser(description='Higher-Order Rectified Flow with configurable losses')
parser.add_argument('--loss_config', type=str, default='M1+M2+SC', 
                    choices=['SC', 'M1+SC', 'M1+M2+SC', 'M1+M2+M3+SC'],
                    help='Loss configuration to use')
parser.add_argument('--interpolation', type=str, default='cubic',
                    choices=['cubic', 'exponential'],
                    help='Interpolation method')
parser.add_argument('--batchsize', type=int, default=1000,
                    help='Batch size for training')
parser.add_argument('--epochs', type=int, default=10000,
                    help='Number of training iterations')
parser.add_argument('--lr', type=float, default=5e-3,
                    help='Learning rate')
parser.add_argument('--num_steps', type=int, default=100,
                    help='Number of ODE steps for sampling')
parser.add_argument('--hidden_dim', type=int, default=100,
                    help='Hidden dimension for MLP')
parser.add_argument('--sc_ratio', type=float, default=0.5,
                    help='Ratio of samples to apply self-consistency loss')
parser.add_argument('--lambda_m2', type=float, default=1e-7,
                    help='Weight for second-order loss')
parser.add_argument('--lambda_m3', type=float, default=1e-10,
                    help='Weight for third-order loss')
parser.add_argument('--save_figs', action='store_true',
                    help='Save output figures')
parser.add_argument('--fig_dir', type=str, default='./figures',
                    help='Directory to save figures')
parser.add_argument('--seed', type=int, default=16,
                    help='Random seed for reproducibility')
parser.add_argument('--num_turns', type=int, default=3,
                    help='Number of spiral turns')
parser.add_argument('--M', type=int, default=600,
                    help='Plotting range')
parser.add_argument('--samples', type=int, default=600,
                    help='Number of samples from each distribution')
parser.add_argument('--christoffel', action='store_true',
                    help='Use Christoffel correction')
parser.add_argument('--flow_type', type=str, default='flow_matching', 
                    choices=['flow_matching', 'alpha_flow', 'mean_flow'],
                    help='Flow type: FM/AlphaFlow/MeanFlow')
parser.add_argument('--alpha', type=float, default=0.5,
                    help='Alpha parameter for AlphaFlow')

args = parser.parse_args()

# ==============================
# Set Random Seeds for Reproducibility
# ==============================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Ensure deterministic behavior for CUDA
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set Python hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print(f"Set random seed to: {seed}")

set_seed(args.seed)

# ==============================
# Batch Re-ordering Function
# ==============================
def reorder_batch(src, tgt, model, d):
    v0, _, _ = model.frist_and_second_order_predict(src, torch.zeros_like(d), d)
    v1, _, _ = model.frist_and_second_order_predict(tgt, torch.ones_like(d), d)
    KE_0 = 0.5 * v0 ** 2
    PE_0 = model.pe_model(src, torch.zeros_like(d), d).squeeze(-1)
    KE_1 = 0.5 * v1 ** 2
    PE_1 = model.pe_model(tgt, torch.ones_like(d), d).squeeze(-1)
    error = ((KE_0.sum(dim=1) + PE_0).unsqueeze(1) - (KE_1.sum(dim=1) + PE_1).unsqueeze(0)) ** 2
    # rearrange tgt to minimize error
    # solve linear assignment
    from scipy.optimize import linear_sum_assignment
    _, col_ind = linear_sum_assignment(error.cpu().detach().numpy())

    # reorder tgt
    tgt_reordered = tgt[col_ind]
    return tgt_reordered

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
print(f"Loss configuration: {args.loss_config}")

# ==============================
# Dataset Creation (spiral)
# ==============================
VAR = 0.3  # variance
M = args.M  # plotting range
COMP = 1000  # number of Gaussian components

# Source distribution (π₀): single Gaussian
source_mean = torch.tensor([0.0, 0.0])
source_cov = VAR * torch.eye(2)
initial_model = MultivariateNormal(source_mean, source_cov)
samples_0 = initial_model.sample([args.samples])

# Spiral function
def spiral_function(theta):
    """r(θ) = 5 * (θ*5 + 10)^1.01"""
    return 5 * pow(theta * 5 + 10, 1.01)

# Generate spiral
num_turns = args.num_turns
angles = [k * (2 * np.pi / COMP) for k in range(COMP * num_turns)]
radii = [spiral_function(theta) for theta in angles]
vertices_1 = [[r * np.cos(theta), r * np.sin(theta)] for r, theta in zip(radii, angles)]
vertices_1_tensor = torch.tensor(vertices_1).float().to(device)  # Move to device

# Target distribution (π₁): Gaussian mixture along spiral
target_mix = Categorical(torch.tensor([1 / len(vertices_1) for _ in range(len(vertices_1))]))
target_comp = MultivariateNormal(torch.tensor(vertices_1).float(),
                                 VAR * torch.stack([torch.eye(2) for _ in range(len(vertices_1))]))
target_model = MixtureSameFamily(target_mix, target_comp)
samples_1 = target_model.sample([args.samples])

# Create training pairs
x_0 = samples_0.detach().clone()[torch.randperm(len(samples_0))]
x_1 = samples_1.detach().clone()[torch.randperm(len(samples_1))]
z_pairs = torch.stack([x_0, x_1], dim=1).to(device)

print('Shape of the samples:', samples_0.shape, samples_1.shape)
print(f'Training pairs: {z_pairs.shape}')

# ==============================
# Model Definitions
# ==============================
class MLP(nn.Module):
    def __init__(self, input_dim=2, hidden_num=100):
        super().__init__()
        self.fc1 = nn.Linear(input_dim + 1 + 1, hidden_num, bias=True)
        self.fc2 = nn.Linear(hidden_num, hidden_num, bias=True)
        self.fc3 = nn.Linear(hidden_num, input_dim, bias=True)
        self.act = lambda x: torch.tanh(x)

    def forward(self, x_input, t, d):
        inputs = torch.cat([x_input, t, d], dim=1)
        x = self.fc1(inputs)
        x = self.act(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.fc3(x)
        return x

class MLP_2nd_order(nn.Module):
    def __init__(self, input_dim=2, hidden_num=100):
        super().__init__()
        self.fc1 = nn.Linear(input_dim + input_dim + 1 + 1, hidden_num, bias=True)
        self.fc2 = nn.Linear(hidden_num, hidden_num, bias=True)
        self.fc3 = nn.Linear(hidden_num, input_dim, bias=True)
        self.act = lambda x: torch.tanh(x)

    def forward(self, first_order_input, x_input, t, d):
        inputs = torch.cat([first_order_input, x_input, t, d], dim=1)
        x = self.fc1(inputs)
        x = self.act(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.fc3(x)
        return x

class MLP_3rd_order(nn.Module):
    def __init__(self, input_dim=2, hidden_num=100):
        super().__init__()
        self.fc1 = nn.Linear(input_dim * 3 + 1 + 1, hidden_num, bias=True)
        self.fc2 = nn.Linear(hidden_num, hidden_num, bias=True)
        self.fc3 = nn.Linear(hidden_num, input_dim, bias=True)
        self.act = lambda x: torch.tanh(x)

    def forward(self, second_order_input, first_order_input, x_input, t, d):
        inputs = torch.cat([second_order_input, first_order_input, x_input, t, d], dim=1)
        x = self.fc1(inputs)
        x = self.act(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.fc3(x)
        return x

class MLP_PE_order(nn.Module):
    """PE model: predicts potential energy"""
    def __init__(self, input_dim=2, hidden_num=100):
        super().__init__()
        self.fc1 = nn.Linear(input_dim + 1 + 1, hidden_num, bias=True)
        self.fc2 = nn.Linear(hidden_num, hidden_num, bias=True)
        self.fc3 = nn.Linear(hidden_num, 1, bias=True)  # Output scalar PE
        self.act = nn.Tanh()

    def forward(self, x_input, t, d):
        inputs = torch.cat([x_input, t, d], dim=1)
        x = self.fc1(inputs)
        x = self.act(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.fc3(x)
        return x

# ==============================
# Rectified Flow Class
# ==============================
class RectifiedFlowHigherOrder():
    def __init__(self, model=None, pe_model=None, interpolation='cubic', num_steps=1000):
        self.model = model
        self.first_order_model = self.model[0] if len(model) > 0 else None
        self.second_order_model = self.model[1] if len(model) > 1 else None
        self.third_order_model = self.model[2] if len(model) > 2 else None
        self.pe_model = pe_model  # Add PE model
        self.interpolation = interpolation
        self.N = num_steps

    def get_train_tuple(self, z0, z1):
        batch_size = z1.shape[0]
        
        # Initialize safely
        z_t = torch.zeros_like(z0)
        t = torch.rand((batch_size, 1), device=device)
        first_order_gt = torch.zeros_like(z0)
        second_order_gt = torch.zeros_like(z0) 
        third_order_gt = torch.zeros_like(z0)
        
        if self.interpolation == 'cubic':
            if args.flow_type == 'flow_matching':
                # ✓ WORKS (your baseline)
                t = torch.rand((batch_size, 1), device=device)
                f_t = 3 * t**2 - 2 * t**3
                z_t = (1 - f_t) * z0 + f_t * z1
                first_order_gt = (6 * t - 6 * t**2) * (z1 - z0)
                
            else:  # mean_flow/alpha_flow - CORRECT TIME SAMPLING
                # Sample r ∈ [0,0.8), t ∈ [r,1]
                r = torch.rand((batch_size, 1), device=device) * 0.8
                t = r + torch.rand((batch_size, 1), device=device) * (1.0 - r)  # t ∈ [r,1]
                
                if args.flow_type == 'mean_flow':
                    s = (r + t) * 0.5
                else:  # alpha_flow
                    s = r + args.alpha * (t - r)
                
                # Position on cubic spline
                f_s = 3 * s**2 - 2 * s**3
                z_t = (1 - f_s) * z0 + f_s * z1
                
                # STRAIGHT-LINE VELOCITY (paper correct)
                first_order_gt = (z1 - z0) / (t - r)
                
            # Higher orders only for flow_matching (your working case)
            if args.flow_type == 'flow_matching':
                second_order_gt = (6 - 12 * t) * (z1 - z0)
                third_order_gt = -12 * torch.ones_like(t, device=device) * (z1 - z0)
        else:
            # Exponential interpolation from file 3
            t = (torch.rand((z1.shape[0], 1)) / (1 + 1e-6)).to(device)
            a = 19.9
            b = 0.1

            # alpha_t = e^{(-1/4 a (1-t)^2-1/2 b(1-t))}
            alpha_t = torch.exp(- (1/4) * a * (1-t)**2 - (1/2) * b * (1-t))
            # first order alpha:
            # d alpha_t / dt = alpha_t * 1/2 * (a (1-t) + b)
            first_order_alpha = alpha_t * (1/2) * (a * (1-t) + b)
            # second order alpha:
            # d^2 alpha_t / dt^2 = 1/2 * (alpha_t * (a(1-x)+b)^2 - a alpha_t)
            second_order_alpha = (1/2) * (alpha_t * (a * (1-t) + b)**2 - a * alpha_t)
            # third order alpha:
            third_order_alpha = (1/2) * (first_order_alpha * (a * (1-t) + b)**2 + 
                                        2 * alpha_t * (a * (1-t) + b) * (-a) - 
                                        a * first_order_alpha)

            # beta_t = sqrt{1-alpha^2}
            beta_t = torch.sqrt(1 - alpha_t**2)
            # first order beta
            # d beta_t / dt = (- alpha  / sqrt{1 - alpha^2}) * (d alpha / dt)
            first_order_beta = (- alpha_t / torch.sqrt(1 - alpha_t**2)) * first_order_alpha
            # second order beta
            second_order_beta = (-1 / ((1 - alpha_t**2) * torch.sqrt(1 - alpha_t**2))) * first_order_alpha**2 + first_order_beta * second_order_alpha / first_order_alpha
            # third order beta
            third_order_beta = torch.zeros_like(third_order_alpha)

            z_t = alpha_t * z1 + beta_t * z0
            first_order_gt = first_order_alpha * z1 + first_order_beta * z0
            second_order_gt = second_order_alpha * z1 + second_order_beta * z0
            third_order_gt = third_order_alpha * z1 + third_order_beta * z0

        return z_t, t, first_order_gt, second_order_gt, third_order_gt

    def frist_and_second_order_predict(self, z_t, t, d):
        tmpd = d.clone()
        tmpd[tmpd < (1 / 128)] = 0
        first_order_pred = self.first_order_model(z_t, t, tmpd)
        
        second_order_pred = None
        if self.second_order_model is not None:
            second_order_pred = self.second_order_model(first_order_pred, z_t, t, tmpd)
        
        third_order_pred = None
        if self.third_order_model is not None:
            third_order_pred = self.third_order_model(second_order_pred, first_order_pred, z_t, t, tmpd)
        
        return first_order_pred, second_order_pred, third_order_pred

    @torch.no_grad()
    def sample_ode(self, z0=None, N=None):
        ### NOTE: Use Euler method to sample from the learned flow
        if N is None:
            N = self.N
        dt = 1./N
        traj = [] # to store the trajectory
        z = z0.detach().clone()
        batchsize = z.shape[0]

        traj.append(z.detach().clone())
        for i in range(N):
            t = torch.ones((batchsize, 1), device=device) * i / N  # Fixed: specify device
            zero = torch.zeros_like(t, device=device)  # Fixed: specify device
            first_order_pred, second_order_pred, third_order_pred = self.frist_and_second_order_predict(z, t, zero)
            
            # Higher order update
            z = z.detach().clone() + first_order_pred * dt
            if second_order_pred is not None:
                z += 0.5 * second_order_pred * dt**2
            if third_order_pred is not None:
                z += (1/6) * third_order_pred * dt**3

            traj.append(z.detach().clone())
        
        distances = torch.cdist(z, vertices_1_tensor)  # Use device tensor
        min_distances, _ = torch.min(distances, dim=1) 
        average_min_distance = min_distances.mean().item()
        print("aver_dist", average_min_distance)
        return traj

    @torch.no_grad()
    def new_gt(self, first_order_gt, second_order_gt, third_order_gt, z_t, t, d, flag): 
        tmpd = d.clone() / 2
        f_t, s_t, th_t = self.frist_and_second_order_predict(z_t, t, tmpd)
        
        # Take step with current predictions
        z_tpd = z_t + tmpd * f_t
        if s_t is not None:
            z_tpd += 0.5 * tmpd**2 * s_t
        if th_t is not None:
            z_tpd += (1/6) * tmpd**3 * th_t
            
        f_tpd, s_tpd, th_tpd = self.frist_and_second_order_predict(z_tpd, t + tmpd, tmpd)
        mask = (flag == 1).squeeze()
        # Update based on loss configuration
        if 'M1' in args.loss_config or args.loss_config == 'SC':
            first_order_gt[mask] = ( f_t[mask] + f_tpd[mask] ) / 2
        if 'M2' in args.loss_config and s_t is not None and s_tpd is not None:
            second_order_gt[mask] = ( s_t[mask] + s_tpd[mask] ) / 2
        if 'M3' in args.loss_config and th_t is not None and th_tpd is not None:
            third_order_gt[mask] = ( th_t[mask] + th_tpd[mask] ) / 2

        return first_order_gt, second_order_gt, third_order_gt

# ==============================
# Training Function
# ==============================
def train_rectified_flow(rectified_flow, optimizer, pairs, batchsize, inner_iters):
    loss_curve = []
    
    # Set loss scales based on configuration
    if args.loss_config == 'SC':
        second_order_loss_scale = 0
        third_order_loss_scale = 0
        first_order_loss_scale = 1
    elif args.loss_config == 'M1+SC':
        second_order_loss_scale = 0
        third_order_loss_scale = 0
        first_order_loss_scale = 1
    elif args.loss_config == 'M1+M2+SC':
        second_order_loss_scale = args.lambda_m2
        third_order_loss_scale = 0
        first_order_loss_scale = 1 - second_order_loss_scale
    elif args.loss_config == 'M1+M2+M3+SC':
        second_order_loss_scale = args.lambda_m2
        third_order_loss_scale = args.lambda_m3
        first_order_loss_scale = 1 - second_order_loss_scale - third_order_loss_scale
    
    for i in tqdm(range(inner_iters+1)):
        optimizer.zero_grad()

        progress = i / args.epochs
        λ_geo = 0#1e-2 * (1 - math.cos(progress * math.pi)) / 2
        λ_ke  = 1e-3 * (1 - math.cos(progress * math.pi)) / 2
        λ_energy = 1e-3 * (1 - math.cos(progress * math.pi)) / 2

        indices = torch.randperm(len(pairs))[:batchsize]
        batch = pairs[indices]
        z0 = batch[:, 0].detach().clone().to(device)
        z1 = batch[:, 1].detach().clone().to(device)

        z_t, t, first_order_gt, second_order_gt, third_order_gt = rectified_flow.get_train_tuple(z0=z0, z1=z1)

        d = torch.zeros_like(t, device=device)

        if progress > 0.5 and progress < 0.85:
            z1 = reorder_batch(z0, z1, rectified_flow, d)
            z_t, t, first_order_gt, second_order_gt, third_order_gt = rectified_flow.get_train_tuple(z0, z1)

        flag = torch.zeros_like(t, dtype=torch.int, device=device)
        num_elements = t.numel()
        num_ones = int(num_elements * args.sc_ratio)
        indices = torch.randperm(num_elements, device=device)[:num_ones]
        flag.view(-1)[indices] = 1
        d[flag == 1] = 1 / 2**torch.randint(0, 8, (num_ones,), device=device)

        # Change first_order_gt and second_order_gt to the flag == 1 part
        first_order_gt, second_order_gt, third_order_gt = rectified_flow.new_gt(
            first_order_gt, second_order_gt, third_order_gt, z_t, t, d, flag
        )

        # ==============================
        # Christoffel correction
        # ==============================
        if args.christoffel:
            z_t.requires_grad_(True)

        first_order_pred, second_order_pred, third_order_pred = rectified_flow.frist_and_second_order_predict(z_t, t, d)

        # ==============================
        # Christoffel correction + Energy Conservation
        # ==============================
        L_geo = 0.0
        L_ke = 0.0
        L_energy = 0.0

        if args.christoffel:
            v = first_order_pred                      # [B, d]
            Jv = compute_jacobian(v, z_t)       # [B, d, d]

            # KE-induced metric
            g = compute_metric(Jv, alpha=alpha)

            # Γ[v,v]
            curv = christoffel_correction(v, z_t, alpha=alpha)
            curv = curv / (curv.norm(dim=1, keepdim=True) + 1e-6)

            # Geodesic residual: ∇_v v = Jv v + Γ[v,v]
            adv = torch.bmm(Jv, v.unsqueeze(-1)).squeeze(-1)
            geo_residual = adv + curv

            L_geo = (geo_residual ** 2).sum(dim=1).mean()

            # Kinetic Energy at t: KE(t) = 0.5 * v^T g v
            KE_t = 0.5 * torch.einsum('bi,bij,bj->b', v, g, v)  # [B]
            
            # Potential Energy at t
            if rectified_flow.pe_model is not None:
                PE_t = rectified_flow.pe_model(z_t, t, d).squeeze(-1)  # [B]
                
                # Take a step forward
                z_t_detached = z_t.detach().clone()  # Detach for forward step
                tmpd = d.clone()
                tmpd[tmpd < (1 / 128)] = 0
                
                # Forward step using predictions
                z_tpd = z_t_detached + tmpd * first_order_pred.detach()
                if second_order_pred is not None:
                    z_tpd = z_tpd + 0.5 * tmpd**2 * second_order_pred.detach()
                if third_order_pred is not None:
                    z_tpd = z_tpd + (1/6) * tmpd**3 * third_order_pred.detach()
                
                z_tpd.requires_grad_(True)
                t_plus_d = t + tmpd
                
                # Get velocity at t+d
                v_tpd, s_tpd, th_tpd = rectified_flow.frist_and_second_order_predict(z_tpd, t_plus_d, tmpd)
                Jv_tpd = compute_jacobian(v_tpd, z_tpd)
                g_tpd = compute_metric(Jv_tpd, alpha=alpha)
                
                # Kinetic Energy at t+d
                KE_tpd = 0.5 * torch.einsum('bi,bij,bj->b', v_tpd, g_tpd, v_tpd)  # [B]
                
                # Potential Energy at t+d
                PE_tpd = rectified_flow.pe_model(z_tpd, t_plus_d, tmpd).squeeze(-1)  # [B]
                
                # Total energy conservation loss: ||E(t+d) - E(t)||^2
                E_t = KE_t + PE_t
                E_tpd = KE_tpd + PE_tpd
                L_energy = ((E_tpd - E_t) ** 2).mean()
            
            L_ke = KE_t.mean()

        total_loss = 0
        
        if 'M1' in args.loss_config or args.loss_config == 'SC':
            first_order_loss = (first_order_gt - first_order_pred).abs().pow(2).sum(dim=1)
            first_order_loss_mean = first_order_loss.mean()
            total_loss += first_order_loss_scale * first_order_loss_mean

        # ==============================
        # Add Christoffel and Energy losses
        # ==============================
        if args.christoffel:
            total_loss += λ_geo * L_geo
            total_loss += λ_ke * L_ke
            if rectified_flow.pe_model is not None:
                total_loss += λ_energy * L_energy
        
        if 'M2' in args.loss_config and second_order_pred is not None:
            second_order_loss = (second_order_gt - second_order_pred).abs().pow(2).sum(dim=1)
            second_order_loss_mean = second_order_loss.mean()
            total_loss += second_order_loss_scale * second_order_loss_mean
        
        if 'M3' in args.loss_config and third_order_pred is not None:
            third_order_loss = (third_order_gt - third_order_pred).abs().pow(2).sum(dim=1)
            third_order_loss_mean = third_order_loss.mean()
            total_loss += third_order_loss_scale * third_order_loss_mean

        total_loss.backward()
        optimizer.step()

        loss_curve.append(total_loss.item())
    
    return rectified_flow, loss_curve

# ==============================
# Visualization Functions
# ==============================
@torch.no_grad()
def draw_plot(rectified_flow, z0, z1, N=None):
    traj = rectified_flow.sample_ode(z0=z0, N=N)

    plt.figure(figsize=(8, 4))
    
    # Plot 1: Source, target, and generated samples
    plt.subplot(1, 2, 1)
    plt.xlim(-M, M)
    plt.ylim(-M, M)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    plt.scatter(traj[0][:, 0].cpu().numpy(), traj[0][:, 1].cpu().numpy(), c='#BD8253' , label=r'$\pi_0$', alpha=0.6)
    plt.scatter(z1[:, 0].cpu().numpy(), z1[:, 1].cpu().numpy(), c='#2E59A7', label=r'$\pi_1$', alpha=0.6)
    plt.scatter(traj[-1][:, 0].cpu().numpy(), traj[-1][:, 1].cpu().numpy(), c='#D9A0B3' , label='Generated', alpha=0.6)
    plt.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0), prop={'size': 9})
    plt.title(f'Transfer result\nLoss: {args.loss_config}', fontsize=13)
    
    # Plot 2: Trajectories
    plt.subplot(1, 2, 2)
    plt.xlim(-M, M)
    plt.ylim(-M, M)
    plt.axis('equal')
    
    traj_particles = torch.stack(traj)
    for i in range(min(30, traj_particles.shape[1])):
        plt.plot(traj_particles[:, i, 0].cpu().numpy(), 
                 traj_particles[:, i, 1].cpu().numpy(), linewidth=0.5)
    plt.title('Transport Trajectories', fontsize=13)
    
    plt.tight_layout()
    
    if args.save_figs:
        suffix = "_curv" if args.christoffel else ""
        filename = f"{args.loss_config.replace('+', '_')}_{args.interpolation}_{args.flow_type}{suffix}_spiral_output.png"
        plt.savefig(f"{args.fig_dir}/{filename}", format='png', bbox_inches='tight', dpi=300)
    
    plt.show()

# ==============================
# Main Execution
# ==============================
def main():
    # Create models based on loss configuration
    models = []
    
    # Always need first-order model
    model1 = MLP(input_dim=2, hidden_num=args.hidden_dim).to(device)
    models.append(model1)
    # Add second-order model if needed
    if 'M2' in args.loss_config or 'M3' in args.loss_config:
        model2 = MLP_2nd_order(input_dim=2, hidden_num=args.hidden_dim).to(device)
        models.append(model2)
    # Add third-order model if needed
    if 'M3' in args.loss_config:
        model3 = MLP_3rd_order(input_dim=2, hidden_num=args.hidden_dim).to(device)
        models.append(model3)

    # Create PE model if using Christoffel correction
    pe_model = None
    if args.christoffel:
        pe_model = MLP_PE_order(input_dim=2, hidden_num=args.hidden_dim).to(device)
        print("Created Potential Energy model for energy conservation")
    
    # Create rectified flow
    rectified_flow = RectifiedFlowHigherOrder(
        model=models,
        pe_model=pe_model,  # Pass PE model
        interpolation=args.interpolation,
        num_steps=args.num_steps
    )
    
    # Create optimizer
    all_params = []
    for model in models:
        all_params.extend(list(model.parameters()))
    if pe_model is not None:
        all_params.extend(list(pe_model.parameters()))
    optimizer = torch.optim.Adam(all_params, lr=args.lr)
    
    print(f"\nTraining Configuration:")
    print(f"  Loss: {args.loss_config}")
    print(f"  Interpolation: {args.interpolation}")
    print(f"  Batch size: {args.batchsize}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning rate: {args.lr}")
    print(f"  ODE steps: {args.num_steps}")
    print(f"  SC ratio: {args.sc_ratio}")
    print(f"  λ_M2: {args.lambda_m2}")
    print(f"  λ_M3: {args.lambda_m3}")
    print(f"  Spiral turns: {args.num_turns}")
    print(f"  Plot range: {args.M}")
    print(f"  Samples: {args.samples}")
    print(f"  Christoffel: {args.christoffel}")
    if args.christoffel:
        print(f"  Energy Conservation: Enabled")
    
    # Train the model
    print("\nStarting training...")
    train_start = time.perf_counter()
    rectified_flow, loss_curve = train_rectified_flow(
        rectified_flow=rectified_flow,
        optimizer=optimizer,
        pairs=z_pairs,
        batchsize=args.batchsize,
        inner_iters=args.epochs
    )
    train_time = time.perf_counter() - train_start
    print(f"\nTRAINING TIME: {train_time:.2f} seconds")
    
    # Plot loss curve
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.plot(loss_curve)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(np.log(np.array(loss_curve) + 1e-10))
    plt.xlabel('Iteration')
    plt.ylabel('log(Loss)')
    plt.title('Log Training Loss')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if args.save_figs:
        suffix = "_curv" if args.christoffel else ""
        loss_path = f"{args.fig_dir}/{args.loss_config.replace('+', '_')}_{args.interpolation}_{args.flow_type}{suffix}_loss.png"
        plt.savefig(loss_path, format='png', bbox_inches='tight', dpi=300)
    
    plt.show()
    
    # Visualize results
    print("\nGenerating samples...")
    test_start = time.perf_counter()
    test_z0 = initial_model.sample([600]).to(device)
    test_z1 = target_model.sample([600]).to(device)
    
    draw_plot(
        rectified_flow=rectified_flow,
        z0=test_z0,
        z1=test_z1,
        N=args.num_steps
    )
    test_time = time.perf_counter() - test_start
    print(f"TESTING TIME: {test_time:.2f} seconds")

    # Visualize loss landscape
    suffix = "_curv" if args.christoffel else ""
    landscape_path = f"{args.fig_dir}/landscape_{args.loss_config.replace('+', '_')}{suffix}.png"
    visualize_loss_landscape(
        rectified_flow=rectified_flow,
        pairs=z_pairs,
        args=args,
        device=device,
        save_path=landscape_path,
        resolution=25
    )
    
    print(f"\nExperiment completed!")
    print(f"Figures saved to: {args.fig_dir}")

# ==============================
# Run the experiment
# ==============================
if __name__ == "__main__":
    import os
    os.makedirs(args.fig_dir, exist_ok=True)
    main()
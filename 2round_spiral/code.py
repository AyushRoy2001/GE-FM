import torch
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
parser.add_argument('--loss_config', type=str, default='M1+M2+M3+SC', 
                    choices=['SC', 'M1+SC', 'M1+M2+SC', 'M1+M2+M3+SC'],
                    help='Loss configuration to use')
parser.add_argument('--interpolation', type=str, default='cubic',
                    choices=['cubic', 'exponential'],
                    help='Interpolation method')
parser.add_argument('--batchsize', type=int, default=800,
                    help='Batch size for training')
parser.add_argument('--epochs', type=int, default=1000,
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
parser.add_argument('--seed', type=int, default=42,
                    help='Random seed for reproducibility')
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
    v0, _, _ = model.predict_derivatives(src, torch.zeros_like(d), d)
    v1, _, _ = model.predict_derivatives(tgt, torch.ones_like(d), d)
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
# Dataset Creation (2-round spiral)
# ==============================
VAR = 0.3  # variance
M = 400  # plotting range
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

# Generate 2-round spiral
num_turns = 2
angles = [k * (2 * np.pi / COMP) for k in range(COMP * num_turns)]
radii = [spiral_function(theta) for theta in angles]
vertices_1 = [[r * np.cos(theta), r * np.sin(theta)] for r, theta in zip(radii, angles)]

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

print(f'Dataset shapes: π₀: {samples_0.shape}, π₁: {samples_1.shape}')
print(f'Training pairs: {z_pairs.shape}')

# ==============================
# Model Definitions
# ==============================
class MLP(nn.Module):
    """First-order model: predicts dz/dt"""
    def __init__(self, input_dim=2, hidden_num=100):
        super().__init__()
        self.fc1 = nn.Linear(input_dim + 1 + 1, hidden_num, bias=True)
        self.fc2 = nn.Linear(hidden_num, hidden_num, bias=True)
        self.fc3 = nn.Linear(hidden_num, input_dim, bias=True)
        self.act = nn.Tanh()

    def forward(self, x_input, t, d):
        inputs = torch.cat([x_input, t, d], dim=1)
        x = self.fc1(inputs)
        x = self.act(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.fc3(x)
        return x

class MLP_2nd_order(nn.Module):
    """Second-order model: predicts d²z/dt²"""
    def __init__(self, input_dim=2, hidden_num=100):
        super().__init__()
        self.fc1 = nn.Linear(input_dim + input_dim + 1 + 1, hidden_num, bias=True)
        self.fc2 = nn.Linear(hidden_num, hidden_num, bias=True)
        self.fc3 = nn.Linear(hidden_num, input_dim, bias=True)
        self.act = nn.Tanh()

    def forward(self, first_order_input, x_input, t, d):
        inputs = torch.cat([first_order_input, x_input, t, d], dim=1)
        x = self.fc1(inputs)
        x = self.act(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.fc3(x)
        return x

class MLP_3rd_order(nn.Module):
    """Third-order model: predicts d³z/dt³"""
    def __init__(self, input_dim=2, hidden_num=100):
        super().__init__()
        self.fc1 = nn.Linear(input_dim * 3 + 1 + 1, hidden_num, bias=True)
        self.fc2 = nn.Linear(hidden_num, hidden_num, bias=True)
        self.fc3 = nn.Linear(hidden_num, input_dim, bias=True)
        self.act = nn.Tanh()

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
class RectifiedFlowHigherOrder:
    def __init__(self, models, pe_model=None, interpolation='cubic', num_steps=1000):
        self.models = models
        self.first_order_model = self.models[0] if len(models) > 0 else None
        self.second_order_model = self.models[1] if len(models) > 1 else None
        self.third_order_model = self.models[2] if len(models) > 2 else None
        self.pe_model = pe_model  # Potential energy model
        self.interpolation = interpolation
        self.N = num_steps
        
        # Interpolation parameters for exponential
        self.a = 19.9
        self.b = 0.1
    
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
            
        else:  # exponential
            # α(t) = exp(-1/4 a (1-t)² - 1/2 b (1-t))
            alpha_t = torch.exp(- (1/4) * self.a * (1-t)**2 - (1/2) * self.b * (1-t))
            # Compute derivatives using automatic differentiation
            alpha_t.requires_grad_(True)
            # First derivative
            first_order_alpha = torch.autograd.grad(
                alpha_t, t, grad_outputs=torch.ones_like(alpha_t), create_graph=True
            )[0]
            # Second derivative
            second_order_alpha = torch.autograd.grad(
                first_order_alpha, t, grad_outputs=torch.ones_like(first_order_alpha), create_graph=True
            )[0]
            # Third derivative
            third_order_alpha = torch.autograd.grad(
                second_order_alpha, t, grad_outputs=torch.ones_like(second_order_alpha)
            )[0]

            # β(t) = sqrt(1 - α²)
            beta_t = torch.sqrt(1 - alpha_t**2)
            # Derivatives of β
            first_order_beta = torch.autograd.grad(
                beta_t, t, grad_outputs=torch.ones_like(beta_t), create_graph=True
            )[0]
            second_order_beta = torch.autograd.grad(
                first_order_beta, t, grad_outputs=torch.ones_like(first_order_beta), create_graph=True
            )[0]
            third_order_beta = torch.autograd.grad(
                second_order_beta, t, grad_outputs=torch.ones_like(second_order_beta)
            )[0]
            
            z_t = alpha_t * z1 + beta_t * z0
            first_order_gt = first_order_alpha * z1 + first_order_beta * z0
            second_order_gt = second_order_alpha * z1 + second_order_beta * z0
            third_order_gt = third_order_alpha * z1 + third_order_beta * z0
            
            # Detach for memory efficiency
            alpha_t = alpha_t.detach()
        
        return z_t, t, first_order_gt, second_order_gt, third_order_gt
    
    def predict_derivatives(self, z_t, t, d):
        """Predict derivatives up to specified order"""
        tmpd = d.clone()
        tmpd[tmpd < (1 / 128)] = 0
        
        first_order_pred = self.first_order_model(z_t, t, tmpd)
        
        second_order_pred = None
        if self.second_order_model is not None:
            second_order_pred = self.second_order_model(first_order_pred, z_t, t, tmpd)
        
        third_order_pred = None
        if self.third_order_model is not None:
            third_order_pred = self.third_order_model(
                second_order_pred, first_order_pred, z_t, t, tmpd
            )
        
        return first_order_pred, second_order_pred, third_order_pred
    
    @torch.no_grad()
    def sample_ode(self, z0=None, N=None):
        """Sample using higher-order ODE integration"""
        if N is None:
            N = self.N
        
        dt = 1.0 / N
        traj = []
        z = z0.detach().clone()
        batchsize = z.shape[0]
        
        traj.append(z.detach().clone())
        
        for i in range(N):
            t = torch.ones((batchsize, 1), device=device) * i / N
            zero = torch.zeros_like(t)
            
            # Get predictions
            first_pred, second_pred, third_pred = self.predict_derivatives(z, t, zero)
            
            # Higher-order Taylor expansion
            update = first_pred * dt
            
            if second_pred is not None:
                update += 0.5 * second_pred * dt**2
            
            if third_pred is not None:
                update += (1/6) * third_pred * dt**3
            
            z = z + update
            traj.append(z.detach().clone())
        
        # Compute average distance to target distribution
        distances = torch.cdist(z, torch.tensor(vertices_1, device=device).float())
        min_distances, _ = torch.min(distances, dim=1)
        average_min_distance = min_distances.mean().item()  
        print(f"Average distance to target: {average_min_distance:.4f}")
        return traj, average_min_distance
    
    @torch.no_grad()
    def apply_self_consistency(self, first_order_gt, second_order_gt, third_order_gt, 
                              z_t, t, d, flag, orders_to_update=[0]):
        """Apply self-consistency correction to ground truth derivatives"""
        tmpd = d.clone() / 2
        
        # Get predictions at current point
        f_t, s_t, th_t = self.predict_derivatives(z_t, t, tmpd)
        # Take half-step
        z_tpd = z_t + tmpd * f_t
        if s_t is not None:
            z_tpd += 0.5 * tmpd**2 * s_t
        if th_t is not None:
            z_tpd += (1/6) * tmpd**3 * th_t
        # Get predictions at new point
        f_tpd, s_tpd, th_tpd = self.predict_derivatives(z_tpd, t + tmpd, tmpd)
        mask = (flag == 1).squeeze()
        
        # Update only specified orders
        if 0 in orders_to_update and f_t is not None and f_tpd is not None:
            first_order_gt[mask] = (f_t[mask] + f_tpd[mask]) / 2
        if 1 in orders_to_update and s_t is not None and s_tpd is not None:
            second_order_gt[mask] = (s_t[mask] + s_tpd[mask]) / 2
        if 2 in orders_to_update and th_t is not None and th_tpd is not None:
            third_order_gt[mask] = (th_t[mask] + th_tpd[mask]) / 2
        
        return first_order_gt, second_order_gt, third_order_gt

# ==============================
# Training Function
# ==============================
def train_rectified_flow(rectified_flow, optimizer, pairs, batchsize, total_iters, 
                         loss_config, sc_ratio=0.5, lambda_m2=1e-7, lambda_m3=1e-10):
    """Train rectified flow with specified loss configuration"""
    loss_curve = []
    
    # Parse loss configuration
    use_m1 = 'M1' in loss_config or 'SC' in loss_config  # M1 always used with SC
    use_m2 = 'M2' in loss_config
    use_m3 = 'M3' in loss_config
    use_sc = 'SC' in loss_config
    
    print(f"Training with: M1={use_m1}, M2={use_m2}, M3={use_m3}, SC={use_sc}")
    
    # Determine which orders to update with SC
    sc_orders = []
    if use_m1:
        sc_orders.append(0)
    if use_m2:
        sc_orders.append(1)
    if use_m3:
        sc_orders.append(2)
    
    pbar = tqdm(range(total_iters + 1))
    for i in pbar:
        optimizer.zero_grad()

        progress = i / args.epochs
        λ_geo = 1e-2 * (1 - math.cos(progress * math.pi)) / 2
        λ_ke  = 1e-3 * (1 - math.cos(progress * math.pi)) / 2
        λ_energy = 1e-3 * (1 - math.cos(progress * math.pi)) / 2
        
        # Sample batch
        indices = torch.randperm(len(pairs))[:batchsize]
        batch = pairs[indices]
        z0 = batch[:, 0].detach().clone()
        z1 = batch[:, 1].detach().clone()
        
        # Get ground truth derivatives
        z_t, t, first_order_gt, second_order_gt, third_order_gt = rectified_flow.get_train_tuple(z0, z1)
        
        # Initialize d (step size for SC)
        d = torch.zeros_like(t)

        if progress > 0.4 and progress < 0.8:
            z1 = reorder_batch(z0, z1, rectified_flow, d)
            z_t, t, first_order_gt, second_order_gt, third_order_gt = rectified_flow.get_train_tuple(z0, z1)
        
        # Apply self-consistency if enabled
        if use_sc:
            # Randomly select samples for SC
            flag = torch.zeros_like(t, dtype=torch.int)
            num_elements = t.numel()
            num_sc = int(num_elements * sc_ratio)
            sc_indices = torch.randperm(num_elements)[:num_sc]
            flag[sc_indices] = 1
            
            # Random step sizes: 1/2^k for k = 0..7
            d[flag == 1] = 1 / 2**torch.randint(0, 8, (num_sc,), device=device)
            
            # Apply SC correction
            first_order_gt, second_order_gt, third_order_gt = rectified_flow.apply_self_consistency(
                first_order_gt, second_order_gt, third_order_gt, z_t, t, d, flag, sc_orders
            )

        # ==============================
        # Christoffel correction
        # ==============================
        if args.christoffel:
            z_t.requires_grad_(True)
        
        # Get model predictions
        first_pred, second_pred, third_pred = rectified_flow.predict_derivatives(z_t, t, d)

        # ==============================
        # Christoffel correction + Energy Conservation
        # ==============================
        L_geo = 0.0
        L_ke = 0.0
        L_energy = 0.0
        
        if args.christoffel:
            v = first_pred                      # [B, d]
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
                z_t.requires_grad_(False)  # Detach for forward step
                tmpd = d.clone()
                tmpd[tmpd < (1 / 128)] = 0
                
                # Forward step using predictions
                z_tpd = z_t + tmpd * first_pred
                if second_pred is not None:
                    z_tpd = z_tpd + 0.5 * tmpd**2 * second_pred
                if third_pred is not None:
                    z_tpd = z_tpd + (1/6) * tmpd**3 * third_pred
                
                z_tpd.requires_grad_(True)
                t_plus_d = t + tmpd
                
                # Get velocity at t+d
                v_tpd, _, _ = rectified_flow.predict_derivatives(z_tpd, t_plus_d, tmpd)
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
        
        # Compute losses
        total_loss = 0
        losses_dict = {}
        
        if use_m1 and first_pred is not None:
            m1_loss = ((first_order_gt - first_pred) ** 2).sum(dim=1).mean()
            total_loss += m1_loss
            losses_dict['M1'] = m1_loss.item()

        # ==============================
        # Add Christoffel and Energy losses
        # ==============================
        if args.christoffel:
            total_loss += λ_geo * L_geo
            total_loss += λ_ke * L_ke
            if rectified_flow.pe_model is not None:
                total_loss += λ_energy * L_energy
                losses_dict['Energy'] = L_energy.item() if isinstance(L_energy, torch.Tensor) else L_energy
        
        if use_m2 and second_pred is not None:
            m2_loss = ((second_order_gt - second_pred) ** 2).sum(dim=1).mean()
            total_loss += lambda_m2 * m2_loss
            losses_dict['M2'] = m2_loss.item()
        if use_m3 and third_pred is not None:
            m3_loss = ((third_order_gt - third_pred) ** 2).sum(dim=1).mean()
            total_loss += lambda_m3 * m3_loss
            losses_dict['M3'] = m3_loss.item()
        
        # Backpropagation
        total_loss.backward()
        optimizer.step()
        
        # Update progress bar
        loss_curve.append(total_loss.item())
        if i % 100 == 0:
            desc = f"Loss: {total_loss.item():.6f}"
            if args.christoffel and rectified_flow.pe_model is not None:
                desc += f" | Energy: {L_energy.item() if isinstance(L_energy, torch.Tensor) else L_energy:.6f}"
            pbar.set_description(desc)
    
    return rectified_flow, loss_curve, losses_dict

# ==============================
# Visualization Functions
# ==============================
@torch.no_grad()
def draw_plot(rectified_flow, z0, z1, title, save_path=None, N=None):
    """Visualize transport results"""
    traj, avg_dist = rectified_flow.sample_ode(z0=z0, N=N)
    
    plt.figure(figsize=(8, 4))
    
    # Plot 1: Source, target, and generated samples
    plt.subplot(1, 2, 1)
    plt.xlim(-M, M)
    plt.ylim(-M, M)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    
    plt.scatter(traj[0][:, 0].cpu().numpy(), traj[0][:, 1].cpu().numpy(), 
                c='#BD8253', label=r'$\pi_0$', alpha=0.6)
    plt.scatter(z1[:, 0].cpu().numpy(), z1[:, 1].cpu().numpy(), 
                c='#2E59A7', label=r'$\pi_1$', alpha=0.6)
    plt.scatter(traj[-1][:, 0].cpu().numpy(), traj[-1][:, 1].cpu().numpy(), 
                c='#D9A0B3', label='Generated', alpha=0.6)
    plt.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0), prop={'size': 8})
    plt.title(f'{title}\nAvg Dist: {avg_dist:.2f}', fontsize=12)
    
    # Plot 2: Trajectories
    plt.subplot(1, 2, 2)
    plt.xlim(-M, M)
    plt.ylim(-M, M)
    plt.axis('equal')
    
    traj_particles = torch.stack(traj)
    for i in range(min(30, traj_particles.shape[1])):
        plt.plot(traj_particles[:, i, 0].cpu().numpy(), 
                 traj_particles[:, i, 1].cpu().numpy(), linewidth=0.5)
    plt.title('Transport Trajectories', fontsize=12)
    
    plt.tight_layout()
    
    if save_path and args.save_figs:
        plt.savefig(save_path, format='png', bbox_inches='tight', dpi=300)
    
    plt.show()
    
    return avg_dist

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
        models=models,
        pe_model=pe_model,
        interpolation=args.interpolation,
        num_steps=args.num_steps
    )
    
    # Create optimizer (include PE model parameters if it exists)
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
    print(f"  Samples: {args.samples}")
    print(f"  Christoffel: {args.christoffel}")
    if args.christoffel:
        print(f"  Energy Conservation: Enabled")
    
    # Train the model
    print("\nStarting training...")
    rectified_flow, loss_curve, final_losses = train_rectified_flow(
        rectified_flow=rectified_flow,
        optimizer=optimizer,
        pairs=z_pairs,
        batchsize=args.batchsize,
        total_iters=args.epochs,
        loss_config=args.loss_config,
        sc_ratio=args.sc_ratio,
        lambda_m2=args.lambda_m2,
        lambda_m3=args.lambda_m3
    )
    
    # Print final losses
    print(f"\nFinal losses:")
    for loss_name, loss_val in final_losses.items():
        print(f"  {loss_name}: {loss_val:.6e}")
    
    # Visualize results
    print("\nGenerating samples...")
    test_z0 = initial_model.sample([400]).to(device)
    test_z1 = target_model.sample([400]).to(device)
    
    suffix = "_curv" if args.christoffel else ""
    save_path = f"{args.fig_dir}/{args.loss_config.replace('+', '_')}_{args.flow_type}{suffix}_result.png"
    avg_dist = draw_plot(
        rectified_flow=rectified_flow,
        z0=test_z0,
        z1=test_z1,
        title=f"Loss: {args.loss_config}",
        save_path=save_path,
        N=args.num_steps
    )
    
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
        loss_path = f"{args.fig_dir}/{args.loss_config.replace('+', '_')}_{args.flow_type}{suffix}_loss.png"
        plt.savefig(loss_path, format='png', bbox_inches='tight', dpi=300)

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
    print(f"Average distance to target: {avg_dist:.4f}")
    print(f"Figures saved to: {args.fig_dir}")

# ==============================
# Run the experiment
# ==============================
if __name__ == "__main__":
    import os
    os.makedirs(args.fig_dir, exist_ok=True)
    main()
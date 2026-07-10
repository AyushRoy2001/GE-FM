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
alpha = 1.0

# ==============================
# Configuration and Arguments
# ==============================
parser = argparse.ArgumentParser(description='Higher-Order Rectified Flow for Four-Mode Dataset')
parser.add_argument('--loss_config', type=str, default='M1+M2+SC', 
                    choices=['M1', 'M2', 'SC', 'M1+M2', 'M1+SC', 'M2+SC', 'M1+M2+SC'],
                    help='Loss configuration to use')
parser.add_argument('--interpolation', type=str, default='exponential',
                    choices=['cubic', 'exponential'],
                    help='Interpolation method')
parser.add_argument('--batchsize', type=int, default=2048,
                    help='Batch size for training')
parser.add_argument('--epochs', type=int, default=5000,
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
parser.add_argument('--save_figs', action='store_true',
                    help='Save output figures')
parser.add_argument('--fig_dir', type=str, default='./figures',
                    help='Directory to save figures')
parser.add_argument('--seed', type=int, default=16,
                    help='Random seed for reproducibility')
parser.add_argument('--M', type=int, default=20,
                    help='Plotting range')
parser.add_argument('--samples', type=int, default=800,
                    help='Number of samples from each distribution')
parser.add_argument('--christoffel', action='store_true',
                    help='Use Christoffel correction')
parser.add_argument('--flow_type', type=str, default='flow_matching', 
                    choices=['flow_matching', 'alpha_flow', 'mean_flow'],
                    help='Flow type: FM/AlphaFlow/MeanFlow')
parser.add_argument('--alpha', type=float, default=0.5,
                    help='Alpha parameter for AlphaFlow')
parser.add_argument('--rotation_angle', type=float, default=np.pi/4,
                    help='Rotation angle for target distribution (radians)')
parser.add_argument('--D_0', type=float, default=5.0,
                    help='Radius for source distribution')
parser.add_argument('--D_1', type=float, default=14.0,
                    help='Radius for target distribution')

args = parser.parse_args()

# ==============================
# Set Random Seeds for Reproducibility
# ==============================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print(f"Set random seed to: {seed}")

set_seed(args.seed)

# ==============================
# Batch Re-ordering Function
# ==============================
def reorder_batch(src, tgt, model, d):
    v0, _ = model.frist_and_second_order_predict(src, torch.zeros_like(d), d)
    v1, _ = model.frist_and_second_order_predict(tgt, torch.ones_like(d), d)
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
# Dataset Creation (Four-Mode)
# ==============================
VAR = 0.3  # variance
M = args.M  # plotting range
COMP = 4  # number of Gaussian components (square vertices)

# Source distribution (π₀): square at smaller radius
angles = [k * (2 * np.pi / COMP) for k in range(COMP)]
vertices_0 = [[args.D_0 * np.cos(theta), args.D_0 * np.sin(theta)] for theta in angles]

initial_mix = Categorical(torch.tensor([1/COMP for _ in range(COMP)]))
initial_comp = MultivariateNormal(torch.tensor(vertices_0).float(),
                                  VAR * torch.stack([torch.eye(2) for _ in range(COMP)]))
initial_model = MixtureSameFamily(initial_mix, initial_comp)
samples_0 = initial_model.sample([args.samples])

# Target distribution (π₁): rotated square at larger radius
vertices_1 = [[args.D_1 * np.cos(theta + args.rotation_angle), 
               args.D_1 * np.sin(theta + args.rotation_angle)] for theta in angles]
vertices_1_tensor = torch.tensor(vertices_1).float().to(device)

target_mix = Categorical(torch.tensor([1/COMP for _ in range(COMP)]))
target_comp = MultivariateNormal(torch.tensor(vertices_1).float(),
                                 VAR * torch.stack([torch.eye(2) for _ in range(COMP)]))
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

class MLP_PE_order(nn.Module):
    """PE model: predicts potential energy"""
    def __init__(self, input_dim=2, hidden_num=100):
        super().__init__()
        #self.fc1 = nn.Linear(input_dim + 1 + 1, hidden_num, bias=True)
        self.fc1 = nn.Linear(input_dim + 1, hidden_num, bias=True)
        self.fc2 = nn.Linear(hidden_num, hidden_num, bias=True)
        self.fc3 = nn.Linear(hidden_num, 1, bias=True)
        self.act = nn.Tanh()

    def forward(self, x_input, t, d):
        inputs = torch.cat([x_input, d], dim=1)
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
    def __init__(self, model=None, pe_model=None, interpolation='exponential', num_steps=1000):
        self.model = model
        self.first_order_model = self.model[0] if len(model) > 0 else None
        self.second_order_model = self.model[1] if len(model) > 1 else None
        self.pe_model = pe_model
        self.interpolation = interpolation
        self.N = num_steps

    def get_train_tuple(self, z0, z1):
        batch_size = z1.shape[0]
        
        if self.interpolation == 'cubic':
            if args.flow_type == 'flow_matching':
                t = torch.rand((batch_size, 1), device=device)
                f_t = 3 * t**2 - 2 * t**3
                z_t = (1 - f_t) * z0 + f_t * z1
                first_order_gt = (6 * t - 6 * t**2) * (z1 - z0)
                second_order_gt = (6 - 12 * t) * (z1 - z0)
            else:  # mean_flow/alpha_flow
                r = torch.rand((batch_size, 1), device=device) * 0.8
                t = r + torch.rand((batch_size, 1), device=device) * (1.0 - r)
                
                if args.flow_type == 'mean_flow':
                    s = (r + t) * 0.5
                else:  # alpha_flow
                    s = r + args.alpha * (t - r)
                
                f_s = 3 * s**2 - 2 * s**3
                z_t = (1 - f_s) * z0 + f_s * z1
                first_order_gt = (z1 - z0) / (t - r)
                second_order_gt = torch.zeros_like(first_order_gt)
        else:  # exponential interpolation (default for four-mode)
            t = (torch.rand((z1.shape[0], 1)) / (1 + 1e-6)).to(device)
            a = 19.9
            b = 0.1

            alpha_t = torch.exp(- (1/4) * a * (1-t)**2 - (1/2) * b * (1-t))
            first_order_alpha = alpha_t * (1/2) * (a * (1-t) + b)
            second_order_alpha = (1/2) * (alpha_t * (a * (1-t) + b)**2 - a * alpha_t)

            beta_t = torch.sqrt(1 - alpha_t**2)
            first_order_beta = (- alpha_t / torch.sqrt(1 - alpha_t**2)) * first_order_alpha
            second_order_beta = (- 1 / ((1 - alpha_t**2) * torch.sqrt(1 - alpha_t**2))) * first_order_alpha + first_order_beta * second_order_alpha

            z_t = alpha_t * z1 + beta_t * z0
            first_order_gt = first_order_alpha * z1 + first_order_beta * z0
            second_order_gt = second_order_alpha * z1 + second_order_beta * z0

        return z_t, t, first_order_gt, second_order_gt

    def frist_and_second_order_predict(self, z_t, t, d):
        tmpd = d.clone()
        tmpd[tmpd < (1 / 128)] = 0
        first_order_pred = self.first_order_model(z_t, t, tmpd)
        
        second_order_pred = None
        if self.second_order_model is not None:
            second_order_pred = self.second_order_model(first_order_pred, z_t, t, tmpd)
        
        return first_order_pred, second_order_pred

    @torch.no_grad()
    def sample_ode(self, z0=None, N=None):
        if N is None:
            N = self.N
        dt = 1./N
        traj = []
        z = z0.detach().clone()
        batchsize = z.shape[0]

        traj.append(z.detach().clone())
        for i in range(N):
            t = torch.ones((batchsize, 1), device=device) * i / N
            zero = torch.zeros_like(t, device=device)
            first_order_pred, second_order_pred = self.frist_and_second_order_predict(z, t, zero)
            
            z = z.detach().clone() + first_order_pred * dt
            if second_order_pred is not None and args.loss_config in ['M2', 'M1+M2', 'M2+SC', 'M1+M2+SC']:
                z += 0.5 * second_order_pred * dt**2

            traj.append(z.detach().clone())
        
        distances = torch.cdist(z, vertices_1_tensor)
        min_distances, _ = torch.min(distances, dim=1) 
        average_min_distance = min_distances.mean().item()
        print(f"Average minimum distance to target modes: {average_min_distance:.4f}")
        
        return traj, average_min_distance

    @torch.no_grad()
    def new_gt(self, first_order_gt, second_order_gt, z_t, t, d, flag): 
        tmpd = d.clone() / 2
        f_t, s_t = self.frist_and_second_order_predict(z_t, t, tmpd)
        
        z_tpd = z_t + tmpd * f_t
        if s_t is not None and args.loss_config in ['M2', 'M1+M2', 'M2+SC', 'M1+M2+SC']:
            z_tpd += 0.5 * tmpd**2 * s_t
            
        f_tpd, s_tpd = self.frist_and_second_order_predict(z_tpd, t + tmpd, tmpd)
        mask = (flag == 1).squeeze()
        
        if 'M1' in args.loss_config or args.loss_config == 'SC' or args.loss_config in ['M1+SC', 'M1+M2+SC']:
            first_order_gt[mask] = ( f_t[mask] + f_tpd[mask] ) / 2
        if 'M2' in args.loss_config and s_t is not None and s_tpd is not None and args.loss_config in ['M2', 'M1+M2', 'M2+SC', 'M1+M2+SC']:
            second_order_gt[mask] = ( s_t[mask] + s_tpd[mask] ) / 2

        return first_order_gt, second_order_gt

# ==============================
# Training Function
# ==============================
def train_rectified_flow(rectified_flow, optimizer, pairs, batchsize, inner_iters):
    loss_curve = []
    avg_dist_curve = []
    
    # Set loss scales based on configuration
    if args.loss_config == 'M1':
        second_order_loss_scale = 0
        first_order_loss_scale = 1
    elif args.loss_config == 'M2':
        second_order_loss_scale = 1
        first_order_loss_scale = 0
    elif args.loss_config == 'SC':
        second_order_loss_scale = 0
        first_order_loss_scale = 1
    elif args.loss_config == 'M1+M2':
        second_order_loss_scale = args.lambda_m2
        first_order_loss_scale = 1 - args.lambda_m2
    elif args.loss_config == 'M1+SC':
        second_order_loss_scale = 0
        first_order_loss_scale = 1
    elif args.loss_config == 'M2+SC':
        second_order_loss_scale = 1
        first_order_loss_scale = 0
    elif args.loss_config == 'M1+M2+SC':
        second_order_loss_scale = args.lambda_m2
        first_order_loss_scale = 1 - args.lambda_m2
    
    for i in tqdm(range(inner_iters+1)):
        optimizer.zero_grad()

        progress = i / args.epochs
        λ_geo = 1e-3 * (1 - math.cos(progress * math.pi))  # Reduced to prevent NaN
        λ_ke  = 1e-4 * (1 - math.cos(progress * math.pi))  # Reduced to prevent NaN
        λ_energy = 1e-4 * (1 - math.cos(progress * math.pi))  # Reduced to prevent NaN

        indices = torch.randperm(len(pairs))[:batchsize]
        batch = pairs[indices]
        z0 = batch[:, 0].detach().clone().to(device)
        z1 = batch[:, 1].detach().clone().to(device)

        z_t, t, first_order_gt, second_order_gt = rectified_flow.get_train_tuple(z0=z0, z1=z1)

        d = torch.zeros_like(t, device=device)

        if progress > 0.5 and progress < 0.7:
            z1 = reorder_batch(z0, z1, rectified_flow, d)
            z_t, t, first_order_gt, second_order_gt = rectified_flow.get_train_tuple(z0, z1)

        flag = torch.zeros_like(t, dtype=torch.int, device=device)
        num_elements = t.numel()
        num_ones = int(num_elements * args.sc_ratio)
        indices = torch.randperm(num_elements, device=device)[:num_ones]
        flag.view(-1)[indices] = 1
        d[flag == 1] = 1 / 2**torch.randint(0, 8, (num_ones,), device=device)

        if 'SC' in args.loss_config and args.sc_ratio > 0:
            first_order_gt, second_order_gt = rectified_flow.new_gt(
                first_order_gt, second_order_gt, z_t, t, d, flag
            )

        if args.christoffel:
            z_t.requires_grad_(True)

        first_order_pred, second_order_pred = rectified_flow.frist_and_second_order_predict(z_t, t, d)

        L_geo = torch.tensor(0.0, device=device)
        L_ke = torch.tensor(0.0, device=device)
        L_energy = torch.tensor(0.0, device=device)

        if args.christoffel:
            v = first_order_pred
            Jv = compute_jacobian(v, z_t)
            g = compute_metric(Jv, alpha=alpha)
            curv = christoffel_correction(v, z_t, alpha=alpha)
            # FIX: Add epsilon to prevent division by zero
            curv_norm = curv.norm(dim=1, keepdim=True)
            curv = curv / (curv_norm + 1e-8)
            adv = torch.bmm(Jv, v.unsqueeze(-1)).squeeze(-1)
            geo_residual = adv + curv
            L_geo = (geo_residual ** 2).sum(dim=1).mean()
            KE_t = 0.5 * torch.einsum('bi,bij,bj->b', v, g, v)
            
            if rectified_flow.pe_model is not None:
                PE_t = rectified_flow.pe_model(z_t, t, d).squeeze(-1)
                z_t_detached = z_t.detach().clone()
                tmpd = d.clone()
                tmpd[tmpd < (1 / 128)] = 0
                z_tpd = z_t_detached + tmpd * first_order_pred.detach()
                if second_order_pred is not None and args.loss_config in ['M2', 'M1+M2', 'M2+SC', 'M1+M2+SC']:
                    z_tpd = z_tpd + 0.5 * tmpd**2 * second_order_pred.detach()
                z_tpd.requires_grad_(True)
                t_plus_d = t + tmpd
                v_tpd, s_tpd = rectified_flow.frist_and_second_order_predict(z_tpd, t_plus_d, tmpd)
                Jv_tpd = compute_jacobian(v_tpd, z_tpd)
                g_tpd = compute_metric(Jv_tpd, alpha=alpha)
                KE_tpd = 0.5 * torch.einsum('bi,bij,bj->b', v_tpd, g_tpd, v_tpd)
                PE_tpd = rectified_flow.pe_model(z_tpd, t_plus_d, tmpd).squeeze(-1)
                E_t = KE_t + PE_t
                E_tpd = KE_tpd + PE_tpd
                L_energy = ((E_tpd - E_t) ** 2).mean()
            
            L_ke = KE_t.mean()

        total_loss = torch.tensor(0.0, device=device)
        
        if 'M1' in args.loss_config or 'SC' in args.loss_config:
            first_order_loss = (first_order_gt - first_order_pred).abs().pow(2).sum(dim=1)
            first_order_loss_mean = first_order_loss.mean()
            total_loss = total_loss + first_order_loss_scale * first_order_loss_mean

        if args.christoffel:
            total_loss = total_loss + λ_geo * L_geo
            total_loss = total_loss + λ_ke * L_ke
            if rectified_flow.pe_model is not None:
                total_loss = total_loss + λ_energy * L_energy
        
        if 'M2' in args.loss_config and second_order_pred is not None:
            second_order_loss = (second_order_gt - second_order_pred).abs().pow(2).sum(dim=1)
            second_order_loss_mean = second_order_loss.mean()
            total_loss = total_loss + second_order_loss_scale * second_order_loss_mean

        # Check for NaN
        if torch.isnan(total_loss).any() or torch.isinf(total_loss).any():
            print(f"Warning: NaN or Inf loss detected at iteration {i}")
            # Skip this iteration
            continue

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(optimizer.param_groups[0]['params'], max_norm=1.0)
        optimizer.step()

        loss_curve.append(total_loss.item())
        
        if i % 100 == 0:
            with torch.no_grad():
                test_z0 = initial_model.sample([100]).to(device)
                _, avg_dist = rectified_flow.sample_ode(z0=test_z0, N=args.num_steps)
                avg_dist_curve.append(avg_dist)
    
    return rectified_flow, loss_curve, avg_dist_curve

# ==============================
# Visualization Functions
# ==============================
@torch.no_grad()
def draw_plot(rectified_flow, z0, z1, N=None):
    traj, avg_dist = rectified_flow.sample_ode(z0=z0, N=N)

    plt.figure(figsize=(12, 4))
    
    # Plot 1: Source, target, and generated samples
    plt.subplot(1, 3, 1)
    plt.xlim(-M, M)
    plt.ylim(-M, M)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.scatter(traj[0][:, 0].cpu().numpy(), traj[0][:, 1].cpu().numpy(), 
                c='#BD8253', label=r'$\pi_0$', alpha=0.6, s=30)
    plt.scatter(z1[:, 0].cpu().numpy(), z1[:, 1].cpu().numpy(), 
                c='#2E59A7', label=r'$\pi_1$', alpha=0.6, s=30)
    plt.scatter(traj[-1][:, 0].cpu().numpy(), traj[-1][:, 1].cpu().numpy(), 
                c='#D9A0B3', label='Generated', alpha=0.6, s=30)
    plt.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0), prop={'size': 10})
    plt.title(f'Four-Mode: {args.loss_config}\nAvg Dist: {avg_dist:.4f}', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Trajectories
    plt.subplot(1, 3, 2)
    plt.xlim(-M, M)
    plt.ylim(-M, M)
    plt.axis('equal')
    
    traj_particles = torch.stack(traj)
    for i in range(min(30, traj_particles.shape[1])):
        plt.plot(traj_particles[:, i, 0].cpu().numpy(), 
                 traj_particles[:, i, 1].cpu().numpy(), linewidth=1.2, alpha=0.6)
    plt.title('Transport Trajectories', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Final positions vs target vertices
    plt.subplot(1, 3, 3)
    plt.xlim(-M, M)
    plt.ylim(-M, M)
    
    # Plot target vertices
    vertices_np = vertices_1_tensor.cpu().numpy()
    plt.scatter(vertices_np[:, 0], vertices_np[:, 1], c='red', s=100, marker='*', 
                label='Target Vertices', alpha=0.8)
    plt.scatter(traj[-1][:, 0].cpu().numpy(), traj[-1][:, 1].cpu().numpy(), 
                c='#2E59A7', label='Generated', alpha=0.5, s=30)
    
    # Draw lines from generated points to nearest vertex
    distances = torch.cdist(traj[-1], vertices_1_tensor)
    min_distances, min_indices = torch.min(distances, dim=1)
    
    for i in range(min(50, len(traj[-1]))):
        idx = min_indices[i].item()
        plt.plot([traj[-1][i, 0].cpu().numpy(), vertices_np[idx, 0]],
                 [traj[-1][i, 1].cpu().numpy(), vertices_np[idx, 1]],
                 'k-', alpha=0.1, linewidth=0.5)
    
    plt.legend(loc='upper right', prop={'size': 9})
    plt.title(f'Final Positions\nAvg Dist: {avg_dist:.4f}', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if args.save_figs:
        suffix = "_curv" if args.christoffel else ""
        filename = f"fourmode_{args.loss_config}_{args.interpolation}_{args.flow_type}{suffix}.png"
        os.makedirs(args.fig_dir, exist_ok=True)
        plt.savefig(os.path.join(args.fig_dir, filename), format='png', bbox_inches='tight', dpi=300)
        print(f"Saved figure to {os.path.join(args.fig_dir, filename)}")
    
    plt.show()

# ==============================
# Main Execution
# ==============================
def main():
    # Create models based on loss configuration
    models = []
    
    # Always need first-order model for all configurations
    model1 = MLP(input_dim=2, hidden_num=args.hidden_dim).to(device)
    models.append(model1)
    
    # Add second-order model if needed
    if args.loss_config in ['M2', 'M1+M2', 'M2+SC', 'M1+M2+SC']:
        model2 = MLP_2nd_order(input_dim=2, hidden_num=args.hidden_dim).to(device)
        models.append(model2)

    pe_model = None
    if args.christoffel:
        pe_model = MLP_PE_order(input_dim=2, hidden_num=args.hidden_dim).to(device)
        print("Created Potential Energy model for energy conservation")
    
    rectified_flow = RectifiedFlowHigherOrder(
        model=models,
        pe_model=pe_model,
        interpolation=args.interpolation,
        num_steps=args.num_steps
    )
    
    all_params = []
    for model in models:
        all_params.extend(list(model.parameters()))
    if pe_model is not None:
        all_params.extend(list(pe_model.parameters()))
    optimizer = torch.optim.Adam(all_params, lr=args.lr)
    
    print(f"\nTraining Configuration:")
    print(f"  Loss: {args.loss_config}")
    print(f"  Interpolation: {args.interpolation}")
    print(f"  Flow type: {args.flow_type}")
    print(f"  Batch size: {args.batchsize}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning rate: {args.lr}")
    print(f"  ODE steps: {args.num_steps}")
    print(f"  SC ratio: {args.sc_ratio}")
    print(f"  λ_M2: {args.lambda_m2}")
    print(f"  Source radius: {args.D_0}")
    print(f"  Target radius: {args.D_1}")
    print(f"  Rotation angle: {args.rotation_angle:.2f} rad")
    print(f"  Christoffel: {args.christoffel}")
    if args.christoffel:
        print(f"  Energy Conservation: {pe_model is not None}")
    
    print("\nStarting training...")
    rectified_flow, loss_curve, avg_dist_curve = train_rectified_flow(
        rectified_flow=rectified_flow,
        optimizer=optimizer,
        pairs=z_pairs,
        batchsize=args.batchsize,
        inner_iters=args.epochs
    )
    
    # Plot loss and distance curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.plot(loss_curve)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    if len(loss_curve) > 0:
        plt.plot(np.log(np.array(loss_curve) + 1e-10))
    plt.xlabel('Iteration')
    plt.ylabel('log(Loss)')
    plt.title('Log Training Loss')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    if len(avg_dist_curve) > 0:
        plt.plot(avg_dist_curve)
    plt.xlabel('Checkpoint (x100 iters)')
    plt.ylabel('Average Distance')
    plt.title('Average Distance to Target Modes')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if args.save_figs:
        suffix = "_curv" if args.christoffel else ""
        loss_path = f"fourmode_{args.loss_config}_{args.interpolation}_{args.flow_type}{suffix}_loss.png"
        os.makedirs(args.fig_dir, exist_ok=True)
        plt.savefig(os.path.join(args.fig_dir, loss_path), format='png', bbox_inches='tight', dpi=300)
        print(f"Saved loss plot to {os.path.join(args.fig_dir, loss_path)}")
    
    #plt.show()
    
    # Visualize results
    print("\nGenerating final samples...")
    test_z0 = initial_model.sample([400]).to(device)
    test_z1 = target_model.sample([400]).to(device)
    
    draw_plot(
        rectified_flow=rectified_flow,
        z0=test_z0,
        z1=test_z1,
        N=args.num_steps
    )
    
    print(f"\nExperiment completed!")

# ==============================
# Run the experiment
# ==============================
if __name__ == "__main__":
    main()
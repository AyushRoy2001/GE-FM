import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

torch.manual_seed(42)
np.random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MLP(nn.Module):
    """MLP for velocity, acceleration, and jerk prediction"""
    def __init__(self, input_dim=2, hidden_dim=100, output_dim=2, n_layers=2, network_type='first_order'):
        super().__init__()
        self.network_type = network_type

        if network_type == 'first_order':
            # Input: x (2), t (1), d (1) -> total 4
            layers = [nn.Linear(4, hidden_dim)]
        elif network_type == 'second_order':
            # Input: x (2), velocity (2), t (1), d (1) -> total 6
            layers = [nn.Linear(6, hidden_dim)]
        elif network_type == 'third_order':
            # Input: x (2), acceleration (2), velocity (2), t (1), d (1) -> total 8
            layers = [nn.Linear(8, hidden_dim)]
        else:
            raise ValueError(f"Unknown network type: {network_type}")
        
        layers.append(nn.ReLU())
        
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x, t, d, vel=None, acc=None):
        """ Args:
            x: (batch, 2) - spatial coordinates
            t: (batch,) or scalar - time
            d: (batch,) or scalar - step size
            vel: (batch, 2) - velocity (for second/third-order networks)
            acc: (batch, 2) - acceleration (for third-order network)
        """
        if isinstance(t, torch.Tensor) and t.dim() == 0:
            t = t.item()
        if isinstance(d, torch.Tensor) and d.dim() == 0:
            d = d.item()
        
        if isinstance(t, (int, float)):
            t = torch.full((x.shape[0],), t, device=x.device)
        if isinstance(d, (int, float)):
            d = torch.full((x.shape[0],), d, device=x.device)
        
        t = t.view(-1, 1)
        d = d.view(-1, 1)
        
        if self.network_type == 'first_order':
            inp = torch.cat([x, t, d], dim=1)
        elif self.network_type == 'second_order':
            if vel is None:
                raise ValueError("Second-order network requires velocity input")
            inp = torch.cat([x, vel, t, d], dim=1)
        elif self.network_type == 'third_order':
            if vel is None or acc is None:
                raise ValueError("Third-order network requires velocity and acceleration inputs")
            inp = torch.cat([x, acc, vel, t, d], dim=1)
        
        return self.network(inp)


def vp_schedule_derivatives(t, a=19.9, b=0.1):
    """Compute derivatives up to third order"""
    alpha_t = np.exp(-0.25 * a * (1 - t)**2 - 0.5 * b * (1 - t))
    beta_t = np.sqrt(1 - alpha_t**2)
    
    # First derivatives
    dalpha_dt = alpha_t * (0.5 * a * (1 - t) + 0.5 * b)
    dbeta_dt = -alpha_t * dalpha_dt / beta_t if beta_t > 1e-8 else 0
    
    # Second derivatives
    d2alpha_dt2 = dalpha_dt * (0.5 * a * (1 - t) + 0.5 * b) - alpha_t * 0.5 * a
    d2beta_dt2 = -(dalpha_dt**2 + alpha_t * d2alpha_dt2) / beta_t if beta_t > 1e-8 else 0
    d2beta_dt2 += alpha_t * dalpha_dt * dbeta_dt / (beta_t**2) if beta_t > 1e-8 else 0
    
    # Third derivatives (simplified approximation)
    d3alpha_dt3 = d2alpha_dt2 * (0.5 * a * (1 - t) + 0.5 * b) - 2 * dalpha_dt * 0.5 * a
    d3beta_dt3 = -(2 * dalpha_dt * d2alpha_dt2 + alpha_t * d3alpha_dt3) / beta_t if beta_t > 1e-8 else 0
    
    return alpha_t, beta_t, dalpha_dt, dbeta_dt, d2alpha_dt2, d2beta_dt2, d3alpha_dt3, d3beta_dt3


def create_third_order_datasets():
    """Create datasets for third-order experiments"""
    def round_spin(n_samples=600, n_rounds=2, variance=0.3):
        t = np.random.uniform(0, n_rounds * 2 * np.pi, n_samples)
        r = t * 30 + np.random.randn(n_samples) * np.sqrt(variance)
        x = r * np.cos(t)
        y = r * np.sin(t)
        return np.stack([x, y], axis=1)
    
    def dot_circle(n_samples=600, variance=0.3):
        # Half from center, half from circle
        n_center = n_samples // 2
        n_circle = n_samples - n_center
        
        # Center dot
        center = np.random.randn(n_center, 2) * np.sqrt(variance) * 5
        
        # Circle
        angles = np.random.uniform(0, 2*np.pi, n_circle)
        radius = 200 + np.random.randn(n_circle) * np.sqrt(variance) * 10
        circle = np.stack([radius * np.cos(angles), radius * np.sin(angles)], axis=1)
        
        return np.concatenate([center, circle], axis=0)
    
    datasets = {}
    
    # 2 Round spin
    datasets['2_round'] = {
        'source': round_spin(600, n_rounds=1),
        'target': round_spin(600, n_rounds=2)
    }
    
    # 3 Round spin
    datasets['3_round'] = {
        'source': round_spin(600, n_rounds=1),
        'target': round_spin(600, n_rounds=3)
    }
    
    # Dot-Circle
    datasets['dot_circle'] = {
        'source': dot_circle(600),
        'target': round_spin(600, n_rounds=2)
    }
    
    return datasets


class ThirdOrderHOMOTrainer:
    """Third-order HOMO trainer"""
    def __init__(self, data_source, data_target, config):
        self.x0_data = data_source
        self.x1_data = data_target
        self.config = config

        self.u1 = MLP(hidden_dim=config['hidden_dim'], n_layers=config['n_layers'], network_type='first_order').to(device)
        self.u2 = MLP(hidden_dim=config['hidden_dim'], n_layers=config['n_layers'], network_type='second_order').to(device)
        self.u3 = MLP(hidden_dim=config['hidden_dim'], n_layers=config['n_layers'], network_type='third_order').to(device)

        params = list(self.u1.parameters()) + list(self.u2.parameters()) + list(self.u3.parameters())
        self.optimizer = optim.Adam(params, lr=config['lr'])
    
    def sample_batch(self, batch_size):
        idx0 = np.random.randint(0, len(self.x0_data), batch_size)
        idx1 = np.random.randint(0, len(self.x1_data), batch_size)
        
        x0 = torch.from_numpy(self.x0_data[idx0]).float().to(device)
        x1 = torch.from_numpy(self.x1_data[idx1]).float().to(device)
        
        return x0, x1
    
    def compute_targets(self, x0, x1, t_np):
        results = vp_schedule_derivatives(t_np)
        alpha_t, beta_t, dalpha, dbeta, d2alpha, d2beta, d3alpha, d3beta = results
        
        v_true = dalpha * x0 + dbeta * x1
        a_true = d2alpha * x0 + d2beta * x1
        j_true = d3alpha * x0 + d3beta * x1
        
        return v_true, a_true, j_true
    
    def train_step(self, batch_size, use_m1=True, use_m2=False, use_m3=False, use_sc=False):
        self.optimizer.zero_grad()
        
        x0, x1 = self.sample_batch(batch_size)
        
        t_np = np.random.uniform(0, 1)
        d = np.random.choice([1/128, 1/64, 1/32, 1/16, 1/8, 1/4, 1/2, 1.0])
        
        results = vp_schedule_derivatives(t_np)
        alpha_t, beta_t = results[0], results[1]
        x_t = alpha_t * x0 + beta_t * x1
        
        t = torch.tensor(t_np, device=device)
        d_tensor = torch.tensor(d, device=device)
        
        loss = 0.0
        
        # M1: First order loss
        if use_m1:
            v_true, _, _ = self.compute_targets(x0, x1, t_np)
            v_pred = self.u1(x_t, t, 2*d_tensor)
            loss += torch.mean((v_pred - v_true)**2)
        
        # M2: Second order loss
        if use_m2:
            _, a_true, _ = self.compute_targets(x0, x1, t_np)
            v_pred = self.u1(x_t, t, 2*d_tensor)
            a_pred = self.u2(x_t, t, 2*d_tensor, vel=v_pred)  # Pass velocity
            loss += torch.mean((a_pred - a_true)**2)

        # M3: Third order loss
        if use_m3:
            _, _, j_true = self.compute_targets(x0, x1, t_np)
            v_pred = self.u1(x_t, t, 2*d_tensor)
            a_pred = self.u2(x_t, t, 2*d_tensor, vel=v_pred)  # Pass velocity
            j_pred = self.u3(x_t, t, 2*d_tensor, vel=v_pred, acc=a_pred)  # Pass velocity and acceleration
            loss += torch.mean((j_pred - j_true)**2)
        
        # SC: Self-consistency loss
        if use_sc and d >= 1/128:
            with torch.no_grad():
                s_t = self.u1(x_t, t, d_tensor)
                a_t = self.u2(x_t, t, d_tensor, vel=s_t)  # Pass velocity
                j_t = self.u3(x_t, t, d_tensor, vel=s_t, acc=a_t)  # Pass velocity and acceleration
                x_t_plus_d = x_t + d_tensor * s_t + (d_tensor**2 / 2) * a_t + (d_tensor**3 / 6) * j_t
                s_t_plus_d = self.u1(x_t_plus_d, t + d_tensor, d_tensor)
                v_target = (s_t + s_t_plus_d) / 2
            
            v_pred = self.u1(x_t, t, 2*d_tensor)
            loss += torch.mean((v_pred - v_target)**2)
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def train(self, n_steps):
        config = self.config
        pbar = tqdm(range(n_steps), desc="Training")
        
        for step in pbar:
            loss = self.train_step(
                config['batch_size'],
                use_m1=config['use_m1'],
                use_m2=config['use_m2'],
                use_m3=config.get('use_m3', False),
                use_sc=config['use_sc']
            )
            
            if step % 100 == 0:
                pbar.set_postfix({'loss': f'{loss:.4f}'})
    
    def sample(self, n_samples=600, n_steps=128):
        self.u1.eval()
        self.u2.eval()
        self.u3.eval()
        
        with torch.no_grad():
            idx = np.random.randint(0, len(self.x0_data), n_samples)
            x = torch.from_numpy(self.x0_data[idx]).float().to(device)
            
            d = 1.0 / n_steps
            d_tensor = torch.tensor(d, device=device)
            
            for step in range(n_steps):
                t = step * d
                t_tensor = torch.tensor(t, device=device)
                
                v = self.u1(x, t_tensor, d_tensor)
                a = self.u2(x, t_tensor, d_tensor, vel=v)
                j = self.u3(x, t_tensor, d_tensor, vel=v, acc=a)
                
                x = x + d_tensor * v + (d_tensor**2 / 2) * a + (d_tensor**3 / 6) * j
        
        return x.cpu().numpy()


def plot_figure_3(datasets, save_dir='results/figure3'):
    """Generate Figure 3 - Third-order HOMO experiments"""
    os.makedirs(save_dir, exist_ok=True)
    
    dataset_names = ['2_round', '3_round', 'dot_circle']
    
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    
    for row_idx, dataset_name in enumerate(dataset_names):
        # Configurations to test
        if dataset_name == '2_round':
            configs = [
                ('SC', {'use_m1': False, 'use_m2': False, 'use_m3': False, 'use_sc': True, 'steps': 180}),
                ('M1+SC', {'use_m1': True, 'use_m2': False, 'use_m3': False, 'use_sc': True, 'steps': 1000}),
                ('M1+M2+SC', {'use_m1': True, 'use_m2': True, 'use_m3': False, 'use_sc': True, 'steps': 1000}),
                ('M1+M2+M3+SC', {'use_m1': True, 'use_m2': True, 'use_m3': True, 'use_sc': True, 'steps': 1000}),
            ]
        elif dataset_name == '3_round':
            configs = [
                ('SC', {'use_m1': False, 'use_m2': False, 'use_m3': False, 'use_sc': True, 'steps': 180}),
                ('M1+SC', {'use_m1': True, 'use_m2': False, 'use_m3': False, 'use_sc': True, 'steps': 2000}),
                ('M1+M2+SC', {'use_m1': True, 'use_m2': True, 'use_m3': False, 'use_sc': True, 'steps': 2000}),
                ('M1+M2+M3+SC', {'use_m1': True, 'use_m2': True, 'use_m3': True, 'use_sc': True, 'steps': 2000}),
            ]
        else:  # dot_circle
            configs = [
                ('SC', {'use_m1': False, 'use_m2': False, 'use_m3': False, 'use_sc': True, 'steps': 180}),
                ('M1+SC', {'use_m1': True, 'use_m2': False, 'use_m3': False, 'use_sc': True, 'steps': 10000}),
                ('M1+M2+SC', {'use_m1': True, 'use_m2': True, 'use_m3': False, 'use_sc': True, 'steps': 10000}),
                ('M1+M2+M3+SC', {'use_m1': True, 'use_m2': True, 'use_m3': True, 'use_sc': True, 'steps': 10000}),
            ]

        print(f"\nProcessing {dataset_name} dataset...")
        ds = datasets[dataset_name]
        x0_data = ds['source']
        x1_data = ds['target']
        
        for col_idx, (name, config_dict) in enumerate(configs):
            print(f"  Training {name}...")
            
            config = {
                'hidden_dim': 100,
                'n_layers': 2,
                'batch_size': 800 if dataset_name == '2_round' else (1000 if dataset_name == '3_round' else 1600),
                'lr': 0.005,
                **config_dict
            }
            
            trainer = ThirdOrderHOMOTrainer(x0_data, x1_data, config)
            trainer.train(config['steps'])
            generated = trainer.sample(n_samples=600)

            ax = axes[row_idx, col_idx]
            ax.scatter(x0_data[:, 0], x0_data[:, 1], c='brown', s=10, alpha=0.6, label='π₀')
            ax.scatter(x1_data[:, 0], x1_data[:, 1], c='indigo', s=10, alpha=0.6, label='π₁')
            ax.scatter(generated[:, 0], generated[:, 1], c='pink', s=10, alpha=0.6, label='Generated')
            
            if dataset_name in ['2_round', 'dot_circle']:
                ax.set_xlim(-500, 500)
                ax.set_ylim(-500, 500)
            else:  # 3_round
                ax.set_xlim(-800, 800)
                ax.set_ylim(-800, 800)
            
            ax.set_aspect('equal')
            ax.legend(fontsize=8)

            ds_label = {'2_round': '2 Round', '3_round': '3 Round', 'dot_circle': 'DC'}[dataset_name]
            ax.set_title(f'({chr(97 + row_idx*4 + col_idx)}) {name} / {ds_label}')
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/figure3.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{save_dir}/figure3.pdf', bbox_inches='tight')
    print(f"\nFigure 3 saved to {save_dir}/figure3.png")
    plt.close()


def main():
    """Generate Figure 3"""
    print("Creating third-order datasets...")
    datasets = create_third_order_datasets()
    
    print("\n" + "="*50)
    print("Generating Figure 3 (Third-Order HOMO)")
    print("="*50)
    plot_figure_3(datasets)
    
    print("\n" + "="*50)
    print("Figure 3 generated successfully!")
    print("="*50)


if __name__ == "__main__":
    main()
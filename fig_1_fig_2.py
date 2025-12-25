import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

torch.manual_seed(42)
np.random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def train_with_sequence(trainer, T_m1, T_m2, T_sc, use_m1, use_m2, use_sc):
    # ---- Stage 1: M1 ----
    if use_m1 or use_m2 or use_sc:
        trainer.config.update({'use_m1': True, 'use_m2': False, 'use_sc': False})
        trainer.train(T_m1)
    # ---- Stage 2: M2 ----
    if use_m2 or use_sc:
        trainer.config.update({'use_m1': False, 'use_m2': True, 'use_sc': False})
        trainer.train(T_m2)
    # ---- Stage 3: SC ----
    if use_sc:
        trainer.config.update({'use_m1': False, 'use_m2': False, 'use_sc': True})
        trainer.train(T_sc)

class MLP(nn.Module):
    """MLP for velocity and acceleration prediction"""
    def __init__(self, input_dim=2, hidden_dim=100, output_dim=2, n_layers=2, network_type='first_order'):
        super().__init__()
        self.network_type = network_type
        
        if network_type == 'first_order':
            # Input: x (2), t (1), d (1) -> total 4
            layers = [nn.Linear(4, hidden_dim)]
        elif network_type == 'second_order':
            # Input: x (2), velocity (2), t (1), d (1) -> total 6
            layers = [nn.Linear(6, hidden_dim)]
        else:
            raise ValueError(f"Unknown network type: {network_type}")
        
        layers.append(nn.ReLU())
        
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x, t, d, vel=None):
        """ Args:
            x: (batch, 2) - spatial coordinates
            t: (batch,) or scalar - time
            d: (batch,) or scalar - step size
            vel: (batch, 2) - velocity (only for second-order network)
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
        
        return self.network(inp)


class GaussianMixtureDataset:
    def __init__(self, n_modes, distance_source, distance_target, variance=0.3, samples_per_mode=100):
        self.n_modes = n_modes
        self.distance_source = distance_source
        self.distance_target = distance_target
        self.variance = variance
        self.samples_per_mode = samples_per_mode
        
    def sample_source(self):
        """Sample source distribution at distance_source"""
        return self._sample_distribution(self.distance_source)
        
    def sample_target(self):
        """Sample target distribution at distance_target"""
        return self._sample_distribution(self.distance_target)
        
    def _sample_distribution(self, distance):
        samples = []
        labels = []
        for i in range(self.n_modes):
            angle = 2 * np.pi * i / self.n_modes
            center = distance * np.array([np.cos(angle), np.sin(angle)])
            mode_samples = np.random.randn(self.samples_per_mode, 2) * np.sqrt(self.variance)
            mode_samples += center
            samples.append(mode_samples)
            labels.append(np.full(self.samples_per_mode, i))
        return np.concatenate(samples), np.concatenate(labels)

def create_datasets():
    datasets = {}
    # Four-mode Gaussian
    four_mode_ds = GaussianMixtureDataset(4, distance_source=5, distance_target=14, samples_per_mode=200)
    datasets['four_mode'] = {
        'source': four_mode_ds.sample_source(),
        'target': four_mode_ds.sample_target()
    }
    # Five-mode Gaussian
    five_mode_ds = GaussianMixtureDataset(5, distance_source=6, distance_target=13, samples_per_mode=200)
    datasets['five_mode'] = {
        'source': five_mode_ds.sample_source(),
        'target': five_mode_ds.sample_target()
    }
    # Eight-mode Gaussian
    eight_mode_ds = GaussianMixtureDataset(8, distance_source=6, distance_target=13, samples_per_mode=100)
    datasets['eight_mode'] = {
        'source': eight_mode_ds.sample_source(),
        'target': eight_mode_ds.sample_target()
    }
    return datasets

def create_complex_datasets():
    def circle_distribution(n_samples=600, radius=10):
        angles = np.random.uniform(0, 2*np.pi, n_samples)
        r = radius + np.random.randn(n_samples) * np.sqrt(0.3)
        x = r * np.cos(angles)
        y = r * np.sin(angles)
        return np.stack([x, y], axis=1)
    
    datasets = {}
    # Circle dataset
    datasets['circle'] = {
        'source': circle_distribution(600, radius=5),
        'target': circle_distribution(600, radius=12)
    }
    # Spin dataset
    def spin_distribution(n_samples=600):
        t = np.random.uniform(0, 6*np.pi, n_samples)
        r = t * 5 + np.random.randn(n_samples) * np.sqrt(0.3)  # Paper: scale factor 5
        x = r * np.cos(t)
        y = r * np.sin(t)
        return np.stack([x, y], axis=1)
    datasets['spin'] = {
        'source': circle_distribution(600, radius=5),
        'target': spin_distribution(600)
    }
    return datasets

def vp_schedule(t, a=19.9, b=0.1):
    """VP ODE schedule from the paper"""
    alpha_t = np.exp(-0.25 * a * (1 - t)**2 - 0.5 * b * (1 - t))
    beta_t = np.sqrt(1 - alpha_t**2)
    return alpha_t, beta_t

def vp_schedule_derivatives(t, a=19.9, b=0.1):
    """Correct derivative calculations (Page 12 of paper)"""
    alpha_t = np.exp(-0.25 * a * (1 - t)**2 - 0.5 * b * (1 - t))
    beta_t = np.sqrt(1 - alpha_t**2 + 1e-8)

    # First derivatives
    dalpha_dt = alpha_t * (0.5 * a * (1 - t) + 0.5 * b)
    if beta_t > 1e-6:
        dbeta_dt = -alpha_t * dalpha_dt / beta_t
    else:
        dbeta_dt = 0.0
    
    # Second derivatives
    d2alpha_dt2 = dalpha_dt * (0.5 * a * (1 - t) + 0.5 * b) - alpha_t * 0.5 * a
    if beta_t > 1e-6:
        d2beta_dt2 = -(dalpha_dt**2 + alpha_t * d2alpha_dt2) / beta_t
    else:
        d2beta_dt2 = 0.0
    
    return alpha_t, beta_t, dalpha_dt, dbeta_dt, d2alpha_dt2, d2beta_dt2

class HOMOTrainer:
    def __init__(self, x0_data, x1_data, src_labels, tgt_labels, config):
        self.x0_data = x0_data
        self.x1_data = x1_data
        self.src_labels = src_labels
        self.tgt_labels = tgt_labels
        self.n_modes = len(np.unique(src_labels))
        self.config = config
        
        self.u1 = MLP(hidden_dim=config['hidden_dim'], n_layers=config['n_layers'], network_type='first_order').to(device)
        self.u2 = MLP(hidden_dim=config['hidden_dim'], n_layers=config['n_layers'], network_type='second_order').to(device)
        
        for m in self.u1.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0.0)
                
        for m in self.u2.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0.0)

        params = list(self.u1.parameters()) + list(self.u2.parameters())
        self.optimizer = optim.Adam(params, lr=config['lr'])

        self.warmup_steps = 100
        
        self.loss_history = []
    
    def sample_batch(self, batch_size):
        modes = np.random.randint(0, self.n_modes, batch_size)
        x0, x1 = [], []

        for m in modes:
            idx0 = np.random.choice(np.where(self.src_labels == m)[0])
            idx1 = np.random.choice(np.where(self.tgt_labels == m)[0])
            x0.append(self.x0_data[idx0])
            x1.append(self.x1_data[idx1])

        x0 = torch.tensor(x0, device=device).float()
        x1 = torch.tensor(x1, device=device).float()
        return x0, x1

    def compute_targets(self, x0, x1, t_np, d):
        a, b, da, db, d2a, d2b = vp_schedule_derivatives(t_np)

        v = da * x0 + db * x1
        a_true = d2a * x0 + d2b * x1
        v_bar = v + 0.5 * d * a_true
        return v_bar, a_true
    
    def train_step(self, batch_size, use_m1=True, use_m2=False, use_sc=False):
        """Single training step"""
        self.optimizer.zero_grad()
        
        x0, x1 = self.sample_batch(batch_size)
        
        t_np = np.random.uniform(0, 1)
        d = np.random.choice([1/128, 1/64, 1/32, 1/16, 1/8, 1/4, 1/2, 1.0])
        
        alpha_t, beta_t = vp_schedule(t_np)
        x_t = alpha_t * x0 + beta_t * x1
        
        t = torch.tensor(t_np, device=device)
        d_tensor = torch.tensor(d, device=device)
        
        loss = torch.tensor(0.0, device=device)
        
        # M1: First order loss
        if use_m1:
            v_true, _ = self.compute_targets(x0, x1, t_np, 2*d)
            v_pred = self.u1(x_t, t, 2*d_tensor)
            loss_m1 = torch.sum((v_pred - v_true)**2) / batch_size
            loss += loss_m1
        # M2: Second order loss
        if use_m2:
            _, a_true = self.compute_targets(x0, x1, t_np, 2*d)
            v_pred = self.u1(x_t, t, 2*d_tensor)
            a_pred = self.u2(x_t, t, 2*d_tensor, vel=v_pred)
            loss_m2 = torch.sum((a_pred - a_true)**2) / batch_size
            loss += 0.5 * loss_m2
        # SC: Self-consistency loss
        if use_sc and d >= 1/128:
            with torch.no_grad():
                s_t = self.u1(x_t, t, d_tensor)
                a_t = self.u2(x_t, t, d_tensor, vel=s_t)
                x_t_plus_d = x_t + d_tensor * s_t + (d_tensor**2 / 2) * a_t
                s_t_plus_d = self.u1(x_t_plus_d, t + d_tensor, d_tensor)
                v_target = (s_t + s_t_plus_d) / 2
            
            v_pred = self.u1(x_t, t, 2*d_tensor)  # MUST require grad
            loss_sc = torch.sum((v_pred - v_target.detach())**2) / batch_size
            loss += 0.5 * loss_sc
        
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.u1.parameters(), max_norm=1.0)
        # torch.nn.utils.clip_grad_norm_(self.u2.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def train(self, n_steps):
        """Train the model"""
        config = self.config
        pbar = tqdm(range(n_steps), desc="Training")
        
        for step in pbar:
            warmup_factor = min(1.0, step / self.warmup_steps)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = config['lr'] * warmup_factor
                
            loss = self.train_step(
                config['batch_size'],
                use_m1=config['use_m1'],
                use_m2=config['use_m2'],
                use_sc=config['use_sc']
            )
            self.loss_history.append(loss)
            
            if step % 100 == 0:
                pbar.set_postfix({'loss': f'{loss:.4f}'})
    
    def sample(self, n_samples=1000, n_steps=128):
        """Generate samples using the trained model"""
        self.u1.eval()
        self.u2.eval()
        
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
                x = x + d_tensor * v + (d_tensor**2 / 2) * a
        
        return x.cpu().numpy()


def compute_euclidean_distance(generated, target):
    """Compute Euclidean distance metric between distributions"""
    # Simple metric: mean minimum distance
    distances = []
    for g_point in generated:
        dists = np.sqrt(np.sum((target - g_point)**2, axis=1))
        distances.append(np.min(dists))
    return np.mean(distances)


def plot_figure_1(datasets, save_dir='results/figure1'):
    """Generate Figure 1 - Gaussian mixture experiments"""
    os.makedirs(save_dir, exist_ok=True)
    
    dataset_name = 'eight_mode'
    ds = datasets[dataset_name]
    
    x0_data, src_labels = ds['source']
    x1_data, tgt_labels = ds['target']
    
    # From Appendix E.2-E.8 - EIGHT MODE SPECIFIC
    configs = [
        ('M1', {'steps': 1000, 'batch_size': 400, 'use_m1': True, 'use_m2': False, 'use_sc': False}),
        ('M2', {'steps': 1000, 'batch_size': 400, 'use_m1': False, 'use_m2': True, 'use_sc': False}),
        ('SC', {'steps': 1000, 'batch_size': 400, 'use_m1': False, 'use_m2': False, 'use_sc': True}),
        ('M1+M2', {'steps': 1000, 'batch_size': 400, 'use_m1': True, 'use_m2': True, 'use_sc': False}),
        ('M1+SC', {'steps': 1000, 'batch_size': 400, 'use_m1': True, 'use_m2': False, 'use_sc': True}),
        ('M2+SC', {'steps': 1000, 'batch_size': 400, 'use_m1': False, 'use_m2': True, 'use_sc': True}),
        ('M1+M2+SC', {'steps': 1000, 'batch_size': 400, 'use_m1': True, 'use_m2': True, 'use_sc': True}),
    ]
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    # Plot original dataset
    ax = axes[0]
    ax.scatter(x0_data[:, 0], x0_data[:, 1], c='brown', s=10, alpha=0.6, label='π₀')
    ax.scatter(x1_data[:, 0], x1_data[:, 1], c='indigo', s=10, alpha=0.6, label='π₁')
    ax.set_xlim(-20, 20)
    ax.set_ylim(-20, 20)
    ax.set_aspect('equal')
    ax.legend()
    ax.set_title('(a) Eight-mode Dataset')
    ax.grid(True, alpha=0.3)
    
    for idx, (name, config_dict) in enumerate(configs, start=1):
        print(f"\nTraining {name}...")
        
        config = {
            'hidden_dim': 200,  # From paper
            'n_layers': 3,      # From paper
            'lr': 0.0001,        # From paper
            **config_dict
        }
        
        trainer = HOMOTrainer(x0_data, x1_data, src_labels, tgt_labels, config)
        T = config['steps']
        train_with_sequence(
            trainer,
            T_m1=T,
            T_m2=T,
            T_sc=T,
            use_m1=config['use_m1'],
            use_m2=config['use_m2'],
            use_sc=config['use_sc']
        )
        
        generated = trainer.sample(n_samples=800)
        
        ax = axes[idx]
        ax.scatter(x0_data[:, 0], x0_data[:, 1], c='brown', s=10, alpha=0.6, label='π₀')
        ax.scatter(x1_data[:, 0], x1_data[:, 1], c='indigo', s=10, alpha=0.6, label='π₁')
        ax.scatter(generated[:, 0], generated[:, 1], c='pink', s=10, alpha=0.6, label='Generated')
        ax.set_xlim(-20, 20)
        ax.set_ylim(-20, 20)
        ax.set_aspect('equal')
        ax.legend()
        ax.set_title(f'({chr(97+idx)}) {name}')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/figure1.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{save_dir}/figure1.pdf', bbox_inches='tight')
    print(f"\nFigure 1 saved to {save_dir}/figure1.png")
    plt.close()


def plot_figure_2(datasets, save_dir='results/figure2'):
    """Generate Figure 2 - Complex distribution experiments"""
    os.makedirs(save_dir, exist_ok=True)
    
    dataset_name = 'spin'
    ds = datasets[dataset_name]
    x0_data = ds['source']
    x1_data = ds['target']
    
    # Configurations to test
    configs = [
        ('M1+M2', {'use_m1': True, 'use_m2': True, 'use_sc': False, 'steps': 1000}),
        ('M1+SC', {'use_m1': True, 'use_m2': False, 'use_sc': True, 'steps': 1000}),
        ('M2+SC', {'use_m1': False, 'use_m2': True, 'use_sc': True, 'steps': 100}),
        ('M1+M2+SC', {'use_m1': True, 'use_m2': True, 'use_sc': True, 'steps': 1000}),
    ]
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    for idx, (name, config_dict) in enumerate(configs):
        print(f"\nTraining {name} for spin dataset...")
        
        config = {
            'hidden_dim': 100,
            'n_layers': 2,
            'batch_size': 1600,
            'lr': 0.005,
            **config_dict
        }
        
        trainer = HOMOTrainer(x0_data, x1_data, config)
        trainer.train(config['steps'])
        
        generated = trainer.sample(n_samples=600)
        
        ax = axes[idx]
        ax.scatter(x0_data[:, 0], x0_data[:, 1], c='brown', s=10, alpha=0.6, label='π₀')
        ax.scatter(x1_data[:, 0], x1_data[:, 1], c='indigo', s=10, alpha=0.6, label='π₁')
        ax.scatter(generated[:, 0], generated[:, 1], c='pink', s=10, alpha=0.6, label='Generated')
        ax.set_xlim(-500, 500)
        ax.set_ylim(-500, 500)
        ax.set_aspect('equal')
        ax.legend()
        ax.set_title(f'({chr(97+idx)}) {name} / spin')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/figure2.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{save_dir}/figure2.pdf', bbox_inches='tight')
    print(f"\nFigure 2 saved to {save_dir}/figure2.png")
    plt.close()


def main():
    """Main function to generate all figures"""
    print("Creating datasets...")
    gaussian_datasets = create_datasets()
    complex_datasets = create_complex_datasets()
    
    print("\n" + "="*50)
    print("Generating Figure 1 (Gaussian Mixtures)")
    print("="*50)
    plot_figure_1(gaussian_datasets)
    
    print("\n" + "="*50)
    print("Generating Figure 2 (Complex Distributions)")
    print("="*50)
    plot_figure_2(complex_datasets)
    
    print("\n" + "="*50)
    print("All figures generated successfully!")
    print("="*50)


if __name__ == "__main__":
    main()
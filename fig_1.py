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

class MLP(nn.Module):
    """MLP for velocity and acceleration prediction"""
    def __init__(self, input_dim=2, hidden_dim=100, output_dim=2, n_layers=2, network_type='first_order'):
        super().__init__()
        self.network_type = network_type
        
        if network_type == 'first_order':
            layers = [nn.Linear(4, hidden_dim)]
        elif network_type == 'second_order':
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
        return self._sample_distribution(self.distance_source)
        
    def sample_target(self):
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
    four_mode_ds = GaussianMixtureDataset(4, distance_source=5, distance_target=14, samples_per_mode=200)
    datasets['four_mode'] = {
        'source': four_mode_ds.sample_source(),
        'target': four_mode_ds.sample_target()
    }
    five_mode_ds = GaussianMixtureDataset(5, distance_source=6, distance_target=13, samples_per_mode=200)
    datasets['five_mode'] = {
        'source': five_mode_ds.sample_source(),
        'target': five_mode_ds.sample_target()
    }
    eight_mode_ds = GaussianMixtureDataset(8, distance_source=6, distance_target=13, samples_per_mode=500)
    datasets['eight_mode'] = {
        'source': eight_mode_ds.sample_source(),
        'target': eight_mode_ds.sample_target()
    }
    return datasets

def vp_schedule(t, a=19.9, b=0.1):
    alpha_t = np.exp(-0.25 * a * (1 - t)**2 - 0.5 * b * (1 - t))
    beta_t = np.sqrt(1 - alpha_t**2)
    return alpha_t, beta_t

def vp_schedule_derivatives(t, a=19.9, b=0.1):
    alpha_t = np.exp(-0.25 * a * (1 - t)**2 - 0.5 * b * (1 - t))
    beta_t = np.sqrt(1 - alpha_t**2 + 1e-8)

    dalpha_dt = alpha_t * (0.5 * a * (1 - t) + 0.5 * b)
    if beta_t > 1e-6:
        dbeta_dt = -alpha_t * dalpha_dt / beta_t
    else:
        dbeta_dt = 0.0
    
    d2alpha_dt2 = dalpha_dt * (0.5 * a * (1 - t) + 0.5 * b) - alpha_t * 0.5 * a
    if beta_t > 1e-6:
        d2beta_dt2 = -(dalpha_dt**2 + alpha_t * d2alpha_dt2) / beta_t
    else:
        d2beta_dt2 = 0.0
    
    return alpha_t, beta_t, dalpha_dt, dbeta_dt, d2alpha_dt2, d2beta_dt2

class HOMOTrainer:
    def __init__(self, x0_data, x1_data, src_labels=None, tgt_labels=None, config=None):
        """Initialize trainer - supports both labeled and unlabeled data"""
        self.x0_data = x0_data
        self.x1_data = x1_data
        self.src_labels = src_labels
        self.tgt_labels = tgt_labels
        self.config = config
        
        self.has_labels = src_labels is not None and tgt_labels is not None
        if self.has_labels:
            self.n_modes = len(np.unique(src_labels))
        else:
            self.n_modes = 1
        
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
        """Sample batch - handles both labeled and unlabeled data"""
        if self.has_labels and self.n_modes > 1:
            # For modal datasets: sample corresponding modes
            modes = np.random.randint(0, self.n_modes, batch_size)
            x0, x1 = [], []

            for m in modes:
                idx0 = np.random.choice(np.where(self.src_labels == m)[0])
                idx1 = np.random.choice(np.where(self.tgt_labels == m)[0])
                x0.append(self.x0_data[idx0])
                x1.append(self.x1_data[idx1])

            x0 = torch.tensor(x0, device=device).float()
            x1 = torch.tensor(x1, device=device).float()
        else:
            # For non-modal datasets: random pairing
            idx0 = np.random.randint(0, len(self.x0_data), batch_size)
            idx1 = np.random.randint(0, len(self.x1_data), batch_size)
            x0 = torch.tensor(self.x0_data[idx0], device=device).float()
            x1 = torch.tensor(self.x1_data[idx1], device=device).float()
        
        return x0, x1

    def compute_targets(self, x0, x1, t_np, d):
        a, b, da, db, d2a, d2b = vp_schedule_derivatives(t_np)
        v = da * x0 + db * x1
        a_true = d2a * x0 + d2b * x1
        v_bar = v + 0.5 * d * a_true
        return v_bar, a_true
    
    def train_step(self, batch_size, use_m1=True, use_m2=False, use_sc=False):
        self.optimizer.zero_grad()
        
        x0, x1 = self.sample_batch(batch_size)
        t_np = np.random.uniform(0, 1)
        d = np.random.choice([1/128, 1/64, 1/32, 1/16, 1/8, 1/4, 1/2, 1.0])
        alpha_t, beta_t = vp_schedule(t_np)
        x_t = alpha_t * x0 + beta_t * x1
        t = torch.tensor(t_np, device=device)
        d_tensor = torch.tensor(d, device=device)
        
        loss = torch.tensor(0.0, device=device)
        
        if use_m1:
            v_true, _ = self.compute_targets(x0, x1, t_np, 2*d)
            v_pred = self.u1(x_t, t, 2*d_tensor)
            loss_m1 = torch.sum((v_pred - v_true)**2) / batch_size
            loss += loss_m1
        if use_m2:
            v_true, a_true = self.compute_targets(x0, x1, t_np, d)
            a_pred = self.u2(x_t, t, d_tensor, vel=v_true)
            loss_m2 = ((a_pred - a_true) ** 2).mean()
            loss += loss_m2
        if use_sc and d >= 1/128:
            with torch.no_grad():
                s_t = self.u1(x_t, t, d_tensor)
                a_t = self.u2(x_t, t, d_tensor, vel=s_t)
                x_t_plus_d = x_t + d_tensor * s_t + 0.5 * (d_tensor ** 2) * a_t
                s_t_plus_d = self.u1(x_t_plus_d, t + d_tensor, d_tensor)
                v_target = 0.5 * (s_t + s_t_plus_d)
            v_pred = self.u1(x_t, t, 2 * d_tensor)
            loss_sc = ((v_pred - v_target.detach()) ** 2).mean()
            loss += loss_sc
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def train(self, n_steps):
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

def compute_distances(generated, target, target_labels=None):
    generated = np.asarray(generated)
    target = np.asarray(target)
    
    euclidean_dist = 0.0
    for g_point in generated:
        dists = np.sqrt(np.sum((target - g_point)**2, axis=1))
        euclidean_dist += np.min(dists)
    euclidean_dist = euclidean_dist / len(generated)
    
    forward_dist = 0.0
    for g_point in generated:
        dists = np.sqrt(np.sum((target - g_point)**2, axis=1))
        forward_dist += np.min(dists)
    forward_dist = forward_dist / len(generated)
    
    backward_dist = 0.0
    for t_point in target:
        dists = np.sqrt(np.sum((generated - t_point)**2, axis=1))
        backward_dist += np.min(dists)
    backward_dist = backward_dist / len(target)
    
    wasserstein_approx = 0.5 * (forward_dist + backward_dist)
    
    coverage_percent = 0.0
    if target_labels is not None and len(np.unique(target_labels)) > 1:
        unique_labels = np.unique(target_labels)
        coverage_count = 0
        
        for label in unique_labels:
            target_mode_points = target[target_labels == label]
            if len(target_mode_points) == 0:
                continue
            for g_point in generated:
                dists = np.sqrt(np.sum((target_mode_points - g_point)**2, axis=1))
                if np.min(dists) < 1.5:
                    coverage_count += 1
                    break
        coverage_percent = (coverage_count / len(unique_labels)) * 100
    
    return euclidean_dist, wasserstein_approx, coverage_percent

def plot_figure_1(datasets, save_dir='results/figure1'):
    os.makedirs(save_dir, exist_ok=True)
    
    dataset_name = 'five_mode'
    ds = datasets[dataset_name]
    x0_data, src_labels = ds['source']
    x1_data, tgt_labels = ds['target']
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.scatter(x0_data[:, 0], x0_data[:, 1], c='brown', s=10, alpha=0.6, label='π₀')
    ax.scatter(x1_data[:, 0], x1_data[:, 1], c='indigo', s=10, alpha=0.6, label='π₁')
    ax.set_xlim(-20, 20)
    ax.set_ylim(-20, 20)
    ax.set_aspect('equal')
    ax.legend()
    ax.set_title('Eight-mode Dataset')
    ax.grid(True, alpha=0.5)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/00_original.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_dir}/00_original.png")
    
    # Eight mode
    # configs = [
    #     ('M1', {'steps': 1000, 'batch_size': 400, 'use_m1': True, 'use_m2': False, 'use_sc': False}),
    #     ('M2', {'steps': 100, 'batch_size': 400, 'use_m1': False, 'use_m2': True, 'use_sc': False}),
    #     ('SC', {'steps': 10, 'batch_size': 400, 'use_m1': False, 'use_m2': False, 'use_sc': True}),
    #     ('M1+M2', {'steps': 1000, 'batch_size': 400, 'use_m1': True, 'use_m2': True, 'use_sc': False}),
    #     ('M1+SC', {'steps': 1000, 'batch_size': 400, 'use_m1': True, 'use_m2': False, 'use_sc': True}),
    #     ('M2+SC', {'steps': 50, 'batch_size': 400, 'use_m1': False, 'use_m2': True, 'use_sc': True}),
    #     ('M1+M2+SC', {'steps': 500, 'batch_size': 400, 'use_m1': True, 'use_m2': True, 'use_sc': True}),
    # ]
    # # Four mode
    # configs = [
    #     ('M1', {'steps': 1000, 'batch_size': 200, 'use_m1': True, 'use_m2': False, 'use_sc': False}),
    #     ('M2', {'steps': 500, 'batch_size': 200, 'use_m1': False, 'use_m2': True, 'use_sc': False}),
    #     ('SC', {'steps': 10, 'batch_size': 200, 'use_m1': False, 'use_m2': False, 'use_sc': True}),
    #     ('M1+M2', {'steps': 1000, 'batch_size': 200, 'use_m1': True, 'use_m2': True, 'use_sc': False}),
    #     ('M1+SC', {'steps': 1000, 'batch_size': 200, 'use_m1': True, 'use_m2': False, 'use_sc': True}),
    #     ('M2+SC', {'steps': 50, 'batch_size': 200, 'use_m1': False, 'use_m2': True, 'use_sc': True}),
    #     ('M1+M2+SC', {'steps': 500, 'batch_size': 200, 'use_m1': True, 'use_m2': True, 'use_sc': True}),
    # ]
    # Five mode
    configs = [
        ('M1', {'steps': 1000, 'batch_size': 300, 'use_m1': True, 'use_m2': False, 'use_sc': False}),
        ('M2', {'steps': 500, 'batch_size': 300, 'use_m1': False, 'use_m2': True, 'use_sc': False}),
        ('SC', {'steps': 10, 'batch_size': 300, 'use_m1': False, 'use_m2': False, 'use_sc': True}),
        ('M1+M2', {'steps': 1000, 'batch_size': 300, 'use_m1': True, 'use_m2': True, 'use_sc': False}),
        ('M1+SC', {'steps': 1000, 'batch_size': 300, 'use_m1': True, 'use_m2': False, 'use_sc': True}),
        ('M2+SC', {'steps': 50, 'batch_size': 300, 'use_m1': False, 'use_m2': True, 'use_sc': True}),
        ('M1+M2+SC', {'steps': 500, 'batch_size': 300, 'use_m1': True, 'use_m2': True, 'use_sc': True}),
    ]

    metrics_file = open(f'{save_dir}/metrics.csv', 'w')
    metrics_file.write("Config,Euclidean,Wasserstein,Coverage%\n")
    
    for idx, (name, config_dict) in enumerate(configs, start=1):
        print(f"\nTraining {name}...")
        # Eight mode
        # config = {
        #     'hidden_dim': 200,
        #     'n_layers': 3,
        #     'lr': 0.0001,
        #     **config_dict
        # }
        # # Four mode
        # config = {
        #     'hidden_dim': 100,
        #     'n_layers': 2,
        #     'lr': 0.0001,
        #     **config_dict
        # }
        # Five mode
        config = {
            'hidden_dim': 200,
            'n_layers': 2,
            'lr': 0.0005,
            **config_dict
        }
        
        trainer = HOMOTrainer(x0_data, x1_data, src_labels, tgt_labels, config)
        trainer.train(config['steps'])
        generated = trainer.sample(n_samples=500)
        
        euclidean, wasserstein, coverage = compute_distances(generated, x1_data, tgt_labels)
        print(f"  Euclidean: {euclidean:.4f}, Wasserstein: {wasserstein:.4f}, Coverage: {coverage:.1f}%")
        metrics_file.write(f"{name},{euclidean:.4f},{wasserstein:.4f},{coverage:.1f}\n")
        
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.scatter(x0_data[:, 0], x0_data[:, 1], c='brown', s=20, alpha=0.7, edgecolors='black', label='π₀')
        ax.scatter(x1_data[:, 0], x1_data[:, 1], c='indigo', s=20, alpha=0.7, edgecolors='black', label='π₁')
        ax.scatter(generated[:, 0], generated[:, 1], c='pink', s=20, alpha=0.7, edgecolors='black', label='Generated')
        ax.set_xlim(-20, 20)
        ax.set_ylim(-20, 20)
        ax.set_aspect('equal')
        ax.legend()
        ax.set_title(f'{name}\nEuclidean: {euclidean:.4f}')
        ax.grid(True, alpha=0.5)

        filename = f'{save_dir}/{idx:02d}_{name.replace("+", "_")}.png'
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {filename}")
    
    metrics_file.close()
    print(f"\nMetrics saved to: {save_dir}/metrics.csv")

def main():
    print("Creating datasets...")
    gaussian_datasets = create_datasets()
    print("Generating Figure 1 (Gaussian Mixtures)")
    plot_figure_1(gaussian_datasets)

if __name__ == "__main__":
    main()

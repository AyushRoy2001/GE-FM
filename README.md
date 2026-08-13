<p align="center">
  <img width="8325" height="2344" alt="method" src="https://github.com/user-attachments/assets/751a4e37-da6f-48ba-8cc1-bd89ed8ba1ac" />
</p>

# [TMLR] GE-FM: Geometry-aware Energy-based Flow Matching for Non-Euclidean Manifolds

<p align="center">
  <strong>*Ayush Roy</strong>¹ &middot; 
  <strong>*Arjun Ramesh Kaushik</strong>¹ &middot; 
  <strong>Vishnu Suresh Lokhande</strong>¹ &middot;
  <strong>Nalini Ratha</strong>¹ &middot;
  <strong>Venu Govindaraju</strong>¹
</p>

<p align="center">
  ¹ University at Buffalo, SUNY (* = equal contribution)
</p>

## Abstract

Flow Matching has emerged as a powerful framework for generative transport and denoising, yet existing formulations are inherently Euclidean, neglecting the curved and time-evolving geometry of diffusion manifolds. Recent higher-order extensions seek to recover curved transport by explicitly modeling higher derivatives, but these approaches introduce instability and accumulate discretization error, particularly in few-step ODE sampling regimes. We propose a strictly first-order-in-time, energy-based flow matching framework that incorporates geometry through Christoffel-adjusted dynamics. Our method defines a total energy as the sum of kinetic energy induced by the predicted velocity field and a learned potential energy, and enforces approximate energy conservation along transport trajectories. Energy conservation encourages optimal low-energy denoising paths and yields a smoother optimization landscape, leading to faster and more stable convergence. Crucially, the energy formulation induces a time-dependent Riemannian metric that captures the evolving diffusion geometry without explicit manifold supervision. Christoffel symbols derived from this induced metric adjust the velocity field to account for curvature, implicitly modeling higher-order effects without introducing additional learnable dynamics. This geometric correction is manifold-agnostic and adapts automatically to the evolving diffusion structure. Empirically, our method outperforms existing few-step baselines, achieving improved performance on both FID and mode coverage ($\sim$ 80\% $\uparrow$ on synthetic spiral datasets). To the best of our knowledge, this is the first geometry-aware flow matching framework that integrates energy conservation and Christoffel dynamics for stable curved generative transport.

## Installation
Clone the repository and run the following commands.

```bash
conda create --name GEFM_synthetic python=3.8
conda activate GEFM_synthetic
pip install -r requirements.txt
```

## Usage
Please run the bash scripts to get the results.
```
bash run_[X].sh
(X = FM to run flow matching for all the curved synthetic datasets,
 X = M1+SC to run M1+SC for all the curved synthetic datasets,
 X = M1+M2+SC to run M1+M2+SC for all the curved  synthetic datasets,
 X = M1+M2+M3+SC to run M1+M2+M3+SC for all the curved synthetic datasets,
 X = gaussian_base run hte baselines, i.e., FM, M1, SC, M2, M3 and all the combinations mentioned in the paper, for all the gaussian synthetic datasets,
 X = gaussian_ours to run GE-FM for all the gaussian synthetic datasets)
```

## Qualitative Results
<img width="2039" height="1292" alt="spiral_1" src="https://github.com/user-attachments/assets/e63199c1-2214-4565-b1b5-60bf1f2d10d1" />

# Citation
```bibtex
@article{royge,
  title={GE-FM: Geometry-aware Energy-based Flow Matching for Non Euclidean Manifolds},
  author={Roy, Ayush and Kaushik, Arjun Ramesh and Lokhande, Vishnu Suresh and Ratha, Nalini K and Govindaraju, Venu},
  journal={Transactions on Machine Learning Research}
}
```

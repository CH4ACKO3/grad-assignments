# Agent Instructions: Experimental Results Generation

This document provides instructions for the agent that will run the experiments and generate results for the CSC6129 Reinforcement Learning assignment. All results should be saved in the `code/result/` directory with specific naming conventions to match the placeholders in the LaTeX report.

## Overview

This assignment requires implementing and comparing various reinforcement learning algorithms on the LunarLander-v2 environment from Gymnasium. The experiments are divided into two main parts:
- **Part I**: Policy Gradient methods (REINFORCE with various improvements)
- **Part II**: Deep Q-Learning methods (DQN, Multi-step, Double DQN)

## Environment Setup

**Environment**: `LunarLander-v2` from Gymnasium
- Install Gymnasium and Box2D: `pip install gymnasium[box2d]` or `uv add gymnasium[box2d]`
- State space: 8-dimensional continuous (position, velocity, angle, angular velocity, leg contacts)
- Action space: 4 discrete actions (do nothing, fire left, fire main, fire right)
- Episode terminates on crash, successful landing, or timeout

## Part I: Policy Gradients

### 1.1 REINFORCE: Trajectory Return vs. Reward-to-Go

**Task**: Compare learning curves for two return computation methods.

**Implementation Requirements**:
- Implement categorical (softmax) policy for discrete actions
- Policy network: Input(8) → Hidden layers → Output(4 logits)
- Implement two return computation methods:
  - Trajectory return: G_t = sum_{t'=0}^{T-1} γ^{t'} r_{t'} (same for all t)
  - Reward-to-go: G_t = sum_{t'=t}^{T-1} γ^{t'-t} r_{t'}

**Hyperparameters** (suggestions):
- Learning rate: 0.001-0.01
- Discount factor γ: 0.99
- Batch size: 4-8 episodes per update
- Training: 100k-300k environment steps
- Random seeds: 3-5 different seeds

**Output**:
- **File**: `code/result/fig_reinforce_return_vs_rtg.png` (or `.pdf`)
- **Content**: Line plot with:
  - X-axis: Environment steps (or episodes)
  - Y-axis: Episode return (average over recent episodes, e.g., moving average with window=10)
  - Two curves: "Trajectory Return" vs "Reward-to-Go"
  - Include standard deviation bands (shaded regions) if using multiple seeds
  - Title: "REINFORCE: Trajectory Return vs. Reward-to-Go"
  - Legend clearly identifying each method

### 1.2 Baselines and Advantage Estimation

**Task**: Compare three configurations for baseline and normalization.

**Implementation Requirements**:
- Baseline: Value network V_φ(s) with same architecture as policy (but output=1)
- Train value network by minimizing MSE: L = ||V_φ(s_t) - G_t||^2
- Advantage: A_t = G_t - V_φ(s_t)
- Advantage normalization: A ← (A - μ) / (σ + ε) within each batch

**Configurations to Compare**:
1. No baseline (pure REINFORCE with reward-to-go)
2. Learned baseline without normalization
3. Learned baseline with normalization

**Hyperparameters**:
- Same as 1.1 for policy
- Value network learning rate: 0.001-0.01 (can be different from policy)
- Multiple value updates per iteration (e.g., 5-10) recommended

**Output**:
- **File**: `code/result/fig_baselines.png` (or `.pdf`)
- **Content**: Line plot with:
  - X-axis: Environment steps
  - Y-axis: Episode return
  - Three curves: "No Baseline", "Baseline (no norm)", "Baseline + Norm"
  - Include standard deviation bands if using multiple seeds
  - Title: "Effect of Baseline and Advantage Normalization"

### 1.3 Generalized Advantage Estimation (GAE)

**Task**: Ablation study over λ ∈ {0, 0.95, 1}.

**Implementation Requirements**:
- Implement GAE: δ_t = r_t + γV_φ(s_{t+1}) - V_φ(s_t)
- A^GAE_t = sum_{l=0}^∞ (γλ)^l δ_{t+l}
- Recursive implementation: A_t = δ_t + γλ A_{t+1} (backward pass)

**Configurations**:
- λ = 0: Pure TD advantage (one-step)
- λ = 0.95: Standard GAE
- λ = 1: Monte Carlo advantage (equivalent to G_t - V_φ(s_t))

**Output**:
- **File**: `code/result/fig_gae_ablation.png` (or `.pdf`)
- **Content**: Line plot with:
  - X-axis: Environment steps
  - Y-axis: Episode return
  - Three curves: "λ=0", "λ=0.95", "λ=1"
  - Include standard deviation bands
  - Title: "GAE Ablation: λ ∈ {0, 0.95, 1}"

**Analysis Questions** (to be answered in report):
1. Why does reward-to-go reduce variance?
2. What failure mode occurs when λ=1? (Look for instability, high variance)

---

## Part II: Deep Q-Learning

### 2.1 DQN Sanity Checks

**Task**: Verify DQN implementation is working correctly.

**Implementation Requirements**:
- Q-network: Input(8) → Hidden layers → Output(4 Q-values)
- Experience replay buffer (capacity: 10k-100k)
- Target network: Copy from online network every K steps (K=1000 suggested)
- ε-greedy exploration: Linear decay from 1.0 → 0.01 over N steps
- Huber loss for TD error
- Gradient clipping (threshold ~10.0)

**TD Target**: y_t = r_t + γ max_{a'} Q̄(s_{t+1}, a')

**Hyperparameters** (suggestions):
- Learning rate: 0.0001-0.001
- Batch size: 32-128
- γ: 0.99
- Buffer capacity: 50000
- Target update frequency K: 1000
- ε decay: 1.0 → 0.01 over 10k-50k steps
- Training: 100k-300k steps

**Output**:
- **File**: `code/result/fig_dqn_sanity.png` (or `.pdf`)
- **Content**: Multi-panel figure with 3 subplots:
  1. **TD Loss over time**: X=steps, Y=TD loss (Huber), should decrease early
  2. **ε schedule**: X=steps, Y=epsilon, should show linear decay
  3. **Episode Return**: X=steps, Y=average episode return (moving average)
- Title: "DQN Sanity Checks"

### 2.2 Multi-step Targets

**Task**: Compare N-step targets for N ∈ {1, 3, 5}.

**Implementation Requirements**:
- N-step target: y^(N)_t = sum_{t'=t}^{t+N-1} γ^{t'-t} r_{t'} + γ^N max_{a'} Q̄(s_{t+N}, a')
- Sample contiguous sequences of length ≥ N from replay buffer
- Handle terminal states: If episode ends before N steps, use shorter return

**Configurations**:
- N=1: Standard DQN
- N=3: Intermediate
- N=5: Longer rollout

**Output**:
- **File**: `code/result/fig_multistep.png` (or `.pdf`)
- **Content**: Line plot with:
  - X-axis: Environment steps
  - Y-axis: Episode return
  - Three curves: "N=1", "N=3", "N=5"
  - Title: "Multi-step Targets: N ∈ {1, 3, 5}"

**Optional Table** (if helpful):
- Final performance comparison: Mean ± std for each N
- Steps to reach threshold (e.g., return > 200)

### 2.3 Double DQN

**Task**: Compare DQN vs. Double DQN.

**Implementation Requirements**:
- Double DQN target: y^DDQN_t = r_t + γ Q̄(s_{t+1}, argmax_{a'} Q(s_{t+1}, a'))
- Action selection: Use online network Q_φ
- Action evaluation: Use target network Q̄_φ̄

**Configurations**:
- Standard DQN
- Double DQN

**Output**:
- **File**: `code/result/fig_dqn_vs_ddqn.png` (or `.pdf`)
- **Content**: Line plot (or multi-panel) with:
  - X-axis: Environment steps
  - Y-axis: Episode return
  - Two curves: "DQN" vs "Double DQN"
  - Optional: Second panel showing average Q-values to visualize overestimation
  - Title: "DQN vs. Double DQN"

**Analysis** (to be discussed in report):
- Do you observe overestimation bias in DQN?
- Does Double DQN show lower Q-values?
- Does it improve stability or final performance?

---

## File Naming and Format Requirements

### Image Format
- **Preferred**: PNG (`.png`) for line plots
- **Alternative**: PDF (`.pdf`) for vector graphics
- Resolution: At least 300 DPI for PNG
- Size: Reasonable dimensions (e.g., 8×6 inches)

### File Names (EXACT names required for LaTeX placeholders)
1. `fig_reinforce_return_vs_rtg.png` (or `.pdf`)
2. `fig_baselines.png` (or `.pdf`)
3. `fig_gae_ablation.png` (or `.pdf`)
4. `fig_dqn_sanity.png` (or `.pdf`)
5. `fig_multistep.png` (or `.pdf`)
6. `fig_dqn_vs_ddqn.png` (or `.pdf`)

### Directory Structure
```
CSC6129-Reinforcement_Learning/
├── code/
│   ├── result/
│   │   ├── fig_reinforce_return_vs_rtg.png
│   │   ├── fig_baselines.png
│   │   ├── fig_gae_ablation.png
│   │   ├── fig_dqn_sanity.png
│   │   ├── fig_multistep.png
│   │   └── fig_dqn_vs_ddqn.png
│   ├── reinforce.py (or similar implementation files)
│   ├── dqn.py
│   └── ... (other code files)
└── tex/
    └── assignment.tex
```

---

## Plotting Guidelines

### General Style
- Use clear, readable fonts (size 12+ for labels)
- Include grid for easier reading
- Use distinct colors for different methods (avoid red-green for colorblind accessibility)
- Include legends in a non-obstructive position
- Label axes clearly with units

### Line Plots
- Use solid lines for main curves
- Use dashed/dotted lines if needed for secondary information
- Add shaded regions (alpha=0.3) for standard deviation bands
- Smooth curves using moving average (window=10-50) for cleaner visualization

### Multi-panel Figures
- Use `matplotlib` subplots with `plt.subplots(nrows, ncols)`
- Ensure consistent axis scales when comparing similar quantities
- Add panel labels (a), (b), (c) if helpful

### Example Python Code Structure
```python
import matplotlib.pyplot as plt
import numpy as np

# Assume data is loaded: steps, returns_method1, returns_method2
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(steps, returns_method1, label='Method 1', linewidth=2)
ax.plot(steps, returns_method2, label='Method 2', linewidth=2)
ax.set_xlabel('Environment Steps', fontsize=12)
ax.set_ylabel('Episode Return', fontsize=12)
ax.set_title('Comparison of Methods', fontsize=14)
ax.legend(fontsize=11)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('code/result/fig_name.png', dpi=300, bbox_inches='tight')
plt.close()
```

---

## Logging and Reproducibility

### Required Logging
- Log episode returns, lengths, and timestamps
- For DQN: Log TD loss, Q-values, epsilon values
- Save logs to CSV or JSON for later analysis
- Log hyperparameters at the start of each run

### Random Seeds
- Use multiple seeds (3-5 recommended) for each experiment
- Set seeds for: Python random, NumPy, PyTorch, Gymnasium environment
```python
import random
import numpy as np
import torch

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
```

### Saving Models (Optional)
- Save checkpoints at regular intervals
- Save final trained models for later analysis
- Use descriptive names: `reinforce_rtg_seed0_final.pth`

---

## Additional Notes for Report

### Hyperparameters to Document
When generating results, document all hyperparameters used:
- Network architecture (layer sizes, activations)
- Learning rates (policy, value, Q-network)
- Batch sizes, buffer capacities
- Discount factor γ, GAE λ, N-step N
- Exploration schedule (ε start/end/decay)
- Gradient clipping threshold
- Training duration (total steps)
- Random seeds used

These should be filled into Section 2.1 "Experimental Details" in the LaTeX report.

### Things NOT to Fill
The following sections in the LaTeX report should be filled AFTER experiments:
- All "Empirical results" placeholders in Analysis chapter
- Answers to report questions (reward-to-go variance, λ=1 failure mode)
- Overestimation bias discussion with empirical findings
- Final conclusions based on experimental comparisons

---

## Validation Checklist

Before considering the task complete, verify:
- [ ] All 6 figure files are generated with correct names
- [ ] Figures are saved in `code/result/` directory
- [ ] Images are high-quality (300 DPI for PNG)
- [ ] All plots have clear labels, legends, titles
- [ ] Learning curves show reasonable learning behavior
- [ ] Multiple seeds used for all experiments (if possible)
- [ ] Hyperparameters are documented (for filling into report)
- [ ] Code runs without errors and can be reproduced

---

## Estimated Compute Requirements

- **Policy Gradients**: ~30-60 minutes per configuration on CPU, faster on GPU
- **DQN methods**: ~1-2 hours per configuration on CPU, ~20-40 minutes on GPU
- **Total**: ~5-10 hours for all experiments with multiple seeds

Consider running experiments in parallel if multiple GPUs/CPUs are available.

---

## Contact and Questions

If any aspect of these instructions is unclear or if unexpected issues arise during implementation:
1. Refer to the original assignment PDF in `prompt/main.pdf`
2. Check standard RL references (Sutton & Barto, Spinning Up in Deep RL)
3. Consult Gymnasium documentation for LunarLander-v2 specifics

Good luck with the experiments!

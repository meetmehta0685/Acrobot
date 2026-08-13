# Acrobot RL Comparison — Repository Rules

## Purpose

This repository is a university reinforcement-learning project comparing tabular
Q-Learning, DQN, and PPO on Gymnasium's `Acrobot-v1` environment.

## Permanent implementation rules

- Implement all reinforcement-learning algorithms in this repository.
- Never introduce Stable-Baselines3, RLlib, CleanRL, or another ready-made RL algorithm.
- Gymnasium is used only for the environment and wrappers; PyTorch is used for neural networks.
- Keep hyperparameters in `config.py` and expose useful overrides through `main.py`.
- Treat `terminated` and `truncated` separately. A time-limit truncation is not a terminal MDP state.
- Seed Python, NumPy, PyTorch, environment action spaces, and evaluation environments.
- Compare algorithms by environment interactions and evaluate them using the same protocol.
- Never fabricate experiment results. Plots and tables must be generated from stored raw data.
- Keep training and evaluation metrics separate.
- Add tests for mathematical components and run them after changes.
- CPU is the default-safe target; CUDA or Apple MPS may be selected automatically.

## Quality bar

Code must be readable by a university student, type annotated where useful, and
document the reinforcement-learning equations represented by non-obvious code.


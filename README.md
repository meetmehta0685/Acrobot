# Acrobot RL Comparison

This project implements and compares tabular Q-Learning, Deep Q-Networks (DQN),
and Proximal Policy Optimization (PPO) from scratch on Gymnasium's
`Acrobot-v1` environment. NumPy is used for tabular learning and analysis;
PyTorch is used for neural networks. No ready-made reinforcement-learning
algorithm library is used.

## Project goal

The goal is to compare how three different reinforcement-learning approaches
learn the same underactuated swing-up task under an equal environment-interaction
budget.

Acrobot has:

- six continuous observations: angle sine/cosine values and two angular velocities;
- three discrete actions: torque `-1`, `0`, or `+1`;
- reward `-1` for each nonterminal step and `0` on successful termination;
- a 500-step time limit.

The implementation treats `terminated` and `truncated` separately. True task
termination does not bootstrap from a next-state value; time-limit truncation
does bootstrap because the underlying task has not succeeded or failed.

## Algorithms

- **Q-Learning:** discretizes the six-dimensional observation into a finite grid
  and applies an epsilon-greedy tabular Bellman update.
- **DQN:** approximates action values with a neural network and uses replay,
  Huber loss, a target network, mini-batches, and gradient clipping.
- **PPO:** learns a categorical actor and value critic from on-policy rollouts,
  using Generalized Advantage Estimation (GAE), advantage normalization, and a
  clipped policy-ratio objective.

The implementations are in [`agents/`](agents/). Algorithm-specific training
loops are in [`training/`](training/); the shared research harness is in
[`experiments/runner.py`](experiments/runner.py).

## Main study

The stored study uses:

1. six tuning trials per method, three tuning-only seeds, and 25,000 interactions
   per trial;
2. five final training seeds and 100,000 interactions per method and seed;
3. greedy evaluation every 5,000 interactions on ten fixed held-out episodes;
4. one-factor sensitivity sweeps for Q-table bins, target synchronization, and
   PPO clipping;
5. ablations removing the DQN target network and PPO clipping; and
6. interquartile means (IQM) with seeded bootstrap confidence intervals.

The study is configured in [`configs/study.yaml`](configs/study.yaml), with the
selected parameters in [`configs/tuned.yaml`](configs/tuned.yaml).

## Results

Under the tested configurations and 100,000-interaction budget, PPO was the only
reliable solver. Return is negative episode length, so values closer to zero are
better.

| Algorithm | Seeds | Final return IQM | 95% CI | Final success IQM | Median steps to solve | Mean training time |
|---|---:|---:|---:|---:|---:|---:|
| PPO | 5 | **-81.7** | **(-87.5, -74.4)** | **100%** | **15,000** | 30.5 s |
| DQN | 5 | -434.7 | (-500.0, -192.2) | 24% | 65,000 | 18.8 s |
| Q-Learning | 5 | -480.5 | (-496.5, -430.6) | 17% | Not solved | **6.6 s** |

These are conditional results for this environment, implementation, search space,
seed set, and interaction budget. They are not a universal ranking of the three
algorithms.

Generated evidence:

- [Learning curves](results/study/figures/learning_curves.png)
- [Seed stability](results/study/figures/seed_stability.png)
- [Hyperparameter sweeps](results/study/figures/hyperparameter_sweeps.png)
- [Ablations](results/study/figures/ablations.png)
- [Computational efficiency](results/study/figures/computational_efficiency.png)
- [Summary table](results/study/tables/main_summary.md) · [CSV](results/study/tables/main_summary.csv)
- [Saved demonstrations](results/study/demos/)

Plots and tables are generated from the raw run records in
[`results/study/`](results/study/); benchmark values are not embedded in the
plotting code.

## Repository structure

```text
agents/                 Q-Learning, DQN, and PPO implementations
analysis/               statistics, tables, figures, and demonstrations
evaluation/             shared exploration-free evaluation protocol
experiments/            study configuration, runner, tuning, and persistence
training/               lightweight algorithm-specific training loops
utils/                  discretization, replay, metrics, plotting, and seeding
configs/                study and tuned hyperparameters
results/study/          raw records, figures, tables, and demonstrations
tests/                  mathematical, component, and integration tests
main.py                simple train/evaluate/compare CLI
run_study.py            complete tuning, matrix, analysis, and demo CLI
colab/                 self-contained portable demonstration
paper/                 research paper and primary-source notes
```

## Installation

Python 3.11 or newer is required. With `uv`:

```bash
uv sync --locked --extra dev
uv run python scripts/smoke_test.py
uv run pytest -q
```

With `pip`:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
pytest -q
```

## Run the project

For a short training run using the simple interface:

```bash
uv run python main.py train --algorithm all --budget 5000 --seed 7
```

For a saved-agent demonstration:

```bash
uv run python main.py demo --algorithm ppo --seed 7
```

For the research study:

```bash
uv run python run_study.py --workers 4 all
```

Completed runs are detected and skipped. Run individual stages with:

```bash
uv run python run_study.py --workers 4 tune
uv run python run_study.py --workers 4 matrix
uv run python run_study.py analyse
```

Use a short diagnostic run to verify the complete research path:

```bash
uv run python run_study.py single --method ppo --seed 7 --steps 5000
```

The headline results are produced by `run_study.py` using the YAML study
configuration. `main.py` is a simpler entry point and uses the central Python
defaults unless its arguments are overridden.

## Testing

The test suite covers:

- discretization boundaries;
- Q-Learning and DQN Bellman targets;
- replay-buffer sampling and overwriting;
- target-network synchronization;
- PPO clipping and GAE;
- termination versus truncation handling;
- ablation switches;
- study persistence and reproducibility; and
- short integration training for all three methods.

Run it with:

```bash
uv run pytest -q
```

## Limitations

- The conclusions are based on one environment, five final seeds, and a finite
  interaction budget.
- Q-Learning's performance depends strongly on the chosen discretization.
- DQN can regress after an intermediate checkpoint; checkpoint selection deserves
  a separate predefined study.
- Wall-clock measurements depend on the local CPU and implementation.
- The evaluation set uses ten fixed episodes per checkpoint, not the environment's
  conventional 100-episode benchmark procedure.

## References

- [Gymnasium Acrobot documentation](https://gymnasium.farama.org/main/environments/classic_control/acrobot/)
- [Watkins and Dayan, Q-Learning](https://doi.org/10.1007/BF00992698)
- [Mnih et al., Human-level control through deep reinforcement learning](https://doi.org/10.1038/nature14236)
- [Schulman et al., Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- [Schulman et al., Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438)

# Acrobot RL Comparison Resources

## Knowledge

- [Acrobot — Gymnasium documentation](https://gymnasium.farama.org/main/environments/classic_control/acrobot/)
  Use for the exact six-dimensional observation, three torque actions, reward,
  target-height termination, and 500-step truncation used by this project.
- [Q-learning — Watkins and Dayan (1992)](https://doi.org/10.1007/BF00992698)
  Use for the original tabular off-policy temporal-difference update and its
  convergence assumptions; do not apply the theorem uncritically to a finite,
  discretized Acrobot experiment.
- [Human-level control through deep reinforcement learning — Mnih et al. (2015)](https://doi.org/10.1038/nature14236)
  Use for DQN's original combination of neural Q-learning, replay, and a target
  network. This repository adapts the idea to a small MLP and Acrobot, not Atari.
- [Proximal Policy Optimization Algorithms — Schulman et al. (2017)](https://arxiv.org/abs/1707.06347)
  Use for PPO's clipped surrogate objective, on-policy rollouts, and repeated
  minibatch optimization.
- [High-Dimensional Continuous Control Using Generalized Advantage Estimation — Schulman et al. (2016)](https://arxiv.org/abs/1506.02438)
  Use for the exponentially weighted TD-residual estimator implemented in
  `agents/ppo.py`.
- [Deep Reinforcement Learning at the Edge of the Statistical Precipice — Agarwal et al. (2021)](https://proceedings.neurips.cc/paper/2021/hash/f514cec81cb148559cf475e7426eed5e-Abstract.html)
  Use for the motivation to report uncertainty and robust aggregates such as
  IQM rather than relying on one run or one point estimate.

## Repository references

- [`FACULTY_DEMO.md`](FACULTY_DEMO.md) — the intended 8–10 minute presentation
  order and viva questions.
- [`paper/RESEARCH_NOTES.md`](paper/RESEARCH_NOTES.md) — claim-by-claim primary
  source notes and citation cautions.
- [`reference/acrobot-rl-comparison.html`](reference/acrobot-rl-comparison.html)
  — the compact code map and algorithm cheat sheet created for this lesson.

## Wisdom

No external community is needed for this lesson. The practical test is to explain
the update rules aloud and defend the interpretation of the repository's own raw
records.

## Gaps

- The repository has strong unit and integration tests, but no interactive visual
  step-through of a single transition. A future lesson could build one around a
  hand-calculated Bellman target and PPO ratio.

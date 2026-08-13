# Primary-source research notes

These notes support the Acrobot comparison paper. They deliberately use original
algorithm papers, the versioned Gymnasium implementation, and the original
statistical-evaluation paper rather than tutorials or secondary summaries.

## Verified claims and appropriate citations

| Paper claim | Primary source | What the source supports |
|---|---|---|
| Tabular Q-learning is an off-policy temporal-difference control method that incrementally estimates optimal action values. Under the paper's discrete-representation and repeated-sampling conditions, the estimates converge to the optimum with probability one. | [Watkins and Dayan (1992)](https://doi.org/10.1007/BF00992698) | Algorithm provenance, the one-step Q-learning update, and the convergence result. Do **not** imply that the theorem directly guarantees convergence after discretising Acrobot with finite data; function/discretisation error and exploration assumptions remain. |
| DQN combines Q-learning with a neural network and uses experience replay and a separate target network to improve training stability. | [Mnih et al. (2015)](https://doi.org/10.1038/nature14236) | Original published DQN system and its two central stabilisation mechanisms. The Nature work evaluates Atari pixels, so the Acrobot multilayer perceptron is an adaptation, not the architecture from the paper. |
| PPO alternates data collection with multiple epochs of stochastic optimisation of a surrogate objective; its clipped objective limits the incentive for large policy-ratio changes. | [Schulman et al. (2017)](https://arxiv.org/abs/1707.06347) | PPO family, clipped surrogate objective, and minibatch/multiple-epoch training. PPO was published as an arXiv report, not as a refereed conference proceedings paper. |
| GAE forms an exponentially weighted sum of temporal-difference residuals and exposes a bias--variance trade-off through lambda. | [Schulman et al. (2016)](https://arxiv.org/abs/1506.02438) | Generalised advantage estimator and its interpretation. Cite the ICLR 2016 paper; the arXiv submission first appeared in 2015. |
| `Acrobot-v1` is a two-link underactuated system with only the inter-link joint actuated. Its observation is six-dimensional, its actions apply torques -1, 0, or +1, and it rewards -1 until terminal success, then 0. The default time limit is 500 steps and the registered reward threshold is -100. | [Gymnasium 1.3.0 Acrobot source](https://github.com/Farama-Foundation/Gymnasium/blob/v1.3.0/gymnasium/envs/classic_control/acrobot.py) and [official documentation](https://gymnasium.farama.org/main/environments/classic_control/acrobot/) | Exact environment interface, reward, terminal predicate, and dynamics implementation. Cite the `v1.3.0` source permalink for reproducibility; the documentation `main` page is mutable. |
| The Acrobot task traces to Sutton's sparse coarse-coding experiments. | [Sutton (1995)](https://proceedings.neurips.cc/paper/1995/hash/8f1d43620bc6bb580df6e80b0dc05c48-Abstract.html) | Historical origin of the benchmark and function-approximation motivation. Gymnasium notes that its default dynamics follow the later book equations; use the Gymnasium source for the exact environment studied here. |
| Few-run deep-RL comparisons should report uncertainty rather than only point estimates; IQM summarizes the middle 50% and is robust to outliers, while bootstrap intervals expose sampling uncertainty. | [Agarwal et al. (2021)](https://proceedings.neurips.cc/paper/2021/hash/f514cec81cb148559cf475e7426eed5e-Abstract.html) | Motivation for IQM and interval estimates. Important scope note: Agarwal et al. primarily formulate aggregate evaluation across multiple tasks and runs. This project has one task and five seeds, so its IQM is across seed returns and its ordinary seed bootstrap is a small-sample uncertainty summary, not a full multi-task `rliable` evaluation. |

## Environment facts for the methods section

- The six observations are `cos(theta1)`, `sin(theta1)`, `cos(theta2)`,
  `sin(theta2)`, angular velocity 1, and angular velocity 2. Velocity bounds are
  +/-4 pi and +/-9 pi radians per second, respectively.
- The discrete action set has three torques: -1, 0, and +1 N m.
- Success occurs when `-cos(theta1) - cos(theta1 + theta2) > 1`.
- Nonterminal transitions give reward -1; the successful terminal transition gives
  reward 0. Thus return is closely related to negative episode length.
- `Acrobot-v1` truncates at 500 steps through Gymnasium's `TimeLimit` wrapper. The
  underlying environment returns termination separately. The paper should explicitly
  state that the implementation does not treat a time-limit truncation as a terminal
  MDP state for value bootstrapping.
- The registered reward threshold is -100. The project consequently uses mean
  evaluation return >= -100 as its solved criterion.

Primary evidence: the [versioned source](https://github.com/Farama-Foundation/Gymnasium/blob/v1.3.0/gymnasium/envs/classic_control/acrobot.py)
and [Acrobot documentation](https://gymnasium.farama.org/main/environments/classic_control/acrobot/).

## BibTeX entries

The entries below are compatible with BibLaTeX/Biber under LuaLaTeX. Conventional
DOIs are included where publishers provide them; arXiv-issued DataCite DOIs are
omitted to avoid presenting preprints as journal articles.

```bibtex
@article{watkins1992qlearning,
  author  = {Watkins, Christopher J. C. H. and Dayan, Peter},
  title   = {Q-Learning},
  journal = {Machine Learning},
  year    = {1992},
  volume  = {8},
  number  = {3--4},
  pages   = {279--292},
  doi     = {10.1007/BF00992698},
  url     = {https://doi.org/10.1007/BF00992698}
}

@article{mnih2015dqn,
  author  = {Mnih, Volodymyr and Kavukcuoglu, Koray and Silver, David and
             Rusu, Andrei A. and Veness, Joel and Bellemare, Marc G. and
             Graves, Alex and Riedmiller, Martin and Fidjeland, Andreas K. and
             Ostrovski, Georg and Petersen, Stig and Beattie, Charles and
             Sadik, Amir and Antonoglou, Ioannis and King, Helen and
             Kumaran, Dharshan and Wierstra, Daan and Legg, Shane and
             Hassabis, Demis},
  title   = {Human-Level Control through Deep Reinforcement Learning},
  journal = {Nature},
  year    = {2015},
  volume  = {518},
  number  = {7540},
  pages   = {529--533},
  doi     = {10.1038/nature14236},
  url     = {https://doi.org/10.1038/nature14236}
}

@article{schulman2017ppo,
  author        = {Schulman, John and Wolski, Filip and Dhariwal, Prafulla and
                   Radford, Alec and Klimov, Oleg},
  title         = {Proximal Policy Optimization Algorithms},
  year          = {2017},
  eprint        = {1707.06347},
  archiveprefix = {arXiv},
  primaryclass  = {cs.LG},
  url           = {https://arxiv.org/abs/1707.06347}
}

@inproceedings{schulman2016gae,
  author    = {Schulman, John and Moritz, Philipp and Levine, Sergey and
               Jordan, Michael I. and Abbeel, Pieter},
  title     = {High-Dimensional Continuous Control Using Generalized Advantage Estimation},
  booktitle = {International Conference on Learning Representations},
  year      = {2016},
  eprint    = {1506.02438},
  archiveprefix = {arXiv},
  primaryclass  = {cs.LG},
  url       = {https://arxiv.org/abs/1506.02438}
}

@inproceedings{agarwal2021statistical,
  author    = {Agarwal, Rishabh and Schwarzer, Max and Castro, Pablo Samuel and
               Courville, Aaron C. and Bellemare, Marc G.},
  title     = {Deep Reinforcement Learning at the Edge of the Statistical Precipice},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2021},
  volume    = {34},
  pages     = {29304--29320},
  url       = {https://proceedings.neurips.cc/paper/2021/hash/f514cec81cb148559cf475e7426eed5e-Abstract.html}
}

@inproceedings{sutton1995generalization,
  author    = {Sutton, Richard S.},
  title     = {Generalization in Reinforcement Learning: Successful Examples Using Sparse Coarse Coding},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {1995},
  volume    = {8},
  publisher = {MIT Press},
  url       = {https://proceedings.neurips.cc/paper/1995/hash/8f1d43620bc6bb580df6e80b0dc05c48-Abstract.html}
}

@software{farama2026gymnasium,
  author  = {{Farama Foundation}},
  title   = {Gymnasium 1.3.0: Acrobot-v1 Environment},
  year    = {2026},
  version = {1.3.0},
  url     = {https://github.com/Farama-Foundation/Gymnasium/blob/v1.3.0/gymnasium/envs/classic_control/acrobot.py},
  urldate = {2026-08-11},
  note    = {Environment implementation and documentation}
}
```

## Citation cautions for the final manuscript

1. Describe DQN and PPO in terms of the implemented components; do not claim an
   exact reproduction of Atari DQN or the original PPO benchmark setup.
2. Separate experimental observations (for example, PPO's result in this repository)
   from claims made by the original algorithm papers.
3. With five final seeds, confidence intervals remain wide. Present them as uncertainty
   estimates, not as proof of statistical superiority.
4. The final-return IQM and “ever solved” rate answer different questions. A method can
   cross the -100 threshold at one checkpoint and later regress; the paper must label
   these metrics precisely.
5. Cite raw result files or the repository commit for every project-specific number.
   Literature references cannot validate locally measured returns or runtimes.

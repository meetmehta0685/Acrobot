# Faculty Demonstration Guide

This project already contains the complete experiment output. Do not retrain all
111 research runs during the presentation; show the saved evidence, then run one
short live demonstration.

## Before entering the room

```bash
cd /Users/meetmehta/Documents/ChatGPT/Acrobot
uv sync --locked --extra dev
uv run pytest -q
uv run python run_study.py analyse
```

Open these files in separate tabs:

1. `README.md`
2. `results/study/figures/learning_curves.png`
3. `results/study/figures/seed_stability.png`
4. `results/study/figures/hyperparameter_sweeps.png`
5. `results/study/figures/ablations.png`
6. `results/study/tables/main_summary.csv`
7. `agents/q_learning.py`, `agents/dqn.py`, and `agents/ppo.py`

Also keep the three GIFs open as a display-safe backup. Test the live Pygame window
on the projector once before the session.

## Recommended 8–10 minute sequence

### 1. Define the problem — 60 seconds

Show the PPO GIF and say:

> Acrobot is an underactuated two-link pendulum: we can torque only the elbow, so
> the controller must build momentum and swing the free end above the target.
> Every step gives -1 reward, making a less-negative return better.

Identify the six continuous observations and the three discrete torque actions.

### 2. Explain the comparison — 90 seconds

- Q-Learning is the interpretable baseline, but needs continuous observations to
  be discretized into a finite table.
- DQN learns Q-values directly from continuous inputs and stabilizes learning with
  replay and a target network.
- PPO learns a stochastic policy and a critic, using GAE and ratio clipping to
  control each policy update.

Open only the central update for each algorithm. Do not scroll through every file.

### 3. Establish experimental fairness — 75 seconds

Say that equal episode counts would be unfair because better policies finish
episodes earlier. Every algorithm therefore receives exactly 100,000 environment
interactions. Hyperparameters were selected on separate tuning seeds. The final
comparison uses five unseen seeds and ten fixed held-out episodes every 5,000
interactions. Evaluation disables exploration.

Mention that the central estimate is the IQM and the final uncertainty bars are
95% bootstrap confidence intervals.

### 4. Present the evidence — 2–3 minutes

Start with `learning_curves.png`:

- PPO crossed the -100 solved threshold quickly and remained stable.
- DQN sometimes learned but frequently regressed, demonstrating seed sensitivity.
- The discretized Q-Learning baseline remained close to the 500-step time limit.

Then show `seed_stability.png` and the summary table:

| Method | Final return IQM | Final success IQM | Interpretation |
|---|---:|---:|---|
| PPO | -81.7 | 100% | Reliable solver in this budget |
| DQN | -434.7 | 24% | Learns on some seeds, unstable overall |
| Q-Learning | -480.5 | 17% | Fast baseline, insufficient representation/budget |

Emphasize “under this configuration and budget,” not “PPO is always best.” Open a
raw CSV briefly to prove each plotted point comes from ten episode returns.

Show the sweep and ablation figures next. The clearest causal observations are
that DQN collapses without its target network and PPO becomes less stable without
clipping. Explain that tuning selected zero entropy coefficient, so an entropy
removal ablation would be identical and was correctly omitted.

### 5. Run the trained controller — 90 seconds

```bash
uv run python run_study.py demo --method ppo
```

This selects the strongest completed PPO seed and uses held-out reset seed 90,000.
If the window fails, immediately show `results/study/demos/ppo_demo.gif`. If there
is time, run DQN using the same command with `--method dqn` and discuss why showing
the best DQN seed does not replace the multi-seed graph.

### 6. Defend correctness — 60 seconds

Show `14 passed`. Mention:

- true terminal transitions do not bootstrap;
- time-limit truncations do bootstrap, because the underlying task did not end;
- PPO cuts GAE at every reset;
- tuning and final evaluation seeds are disjoint;
- matching completed runs are resume-safe and identified by configuration hashes.

### 7. Close with limitations — 45 seconds

State that conclusions are specific to Acrobot, the tested search spaces, five
seeds, and 100,000 interactions. A larger seed count, broader tuning, and a
predefined checkpoint-selection rule would strengthen the final paper.

## Colab fallback

Open the uploaded notebook directly:

https://colab.research.google.com/drive/1MQ2bgt5Xlgs9z_qLQN5eMtxyR5cLOQYR

The Colab notebook is for a self-contained quick demonstration. Use `BUDGET=2000`,
`SEEDS=[0]`, and two evaluation episodes during a live rehearsal. Clearly say that
these short results verify execution only; the README figures come from the full
local research matrix.

## Likely viva questions

- **Why not compare equal episodes?** Episode length is a consequence of policy
  quality, so equal episodes give algorithms different numbers of interactions.
- **Why discretize Q-Learning?** A finite Q-table cannot index an uncountable
  continuous state space.
- **Why does DQN use a target network?** Slowly changing targets reduce the moving-
  target instability in bootstrapped neural value learning.
- **Why replay transitions?** Replay reduces short-range correlation and reuses
  experience for multiple gradient updates.
- **Why is PPO on-policy?** The probability ratio is defined relative to the policy
  that generated the rollout.
- **Why clip PPO's ratio?** Clipping limits the incentive for excessively large
  policy changes from one batch.
- **What exactly counts as success?** `terminated=True` from Gymnasium, meaning the
  target height was reached—not a time-limit truncation.
- **Why IQM instead of only the mean?** IQM is less dominated by a few extreme
  seeds while still using the central half of performance observations.
- **Why did DQN “solve” but finish poorly?** Some seeds crossed the threshold at an
  intermediate checkpoint and later regressed; the final-policy metric records
  that instability.
- **Why is there no entropy ablation?** Equal-budget tuning selected coefficient
  zero, so removing entropy would not alter the trained method.
- **Can this prove PPO is universally superior?** No. It supports a conditional
  conclusion for this environment, search design, and interaction budget.

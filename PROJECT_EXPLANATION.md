# Acrobot Project Viva Notes

## Seven answers to know

1. **What is Acrobot?** A two-link pendulum with one motor at the middle joint.
   Because it has fewer actuators than degrees of freedom, it is underactuated.
2. **State, action, reward:** The state has angle sine/cosine values and angular
   velocities. The action is one of three torques. Reward is -1 per step until
   success, encouraging fast swing-up.
3. **Q-Learning:** It learns action values from Bellman temporal-difference errors
   and acts epsilon-greedily while training.
4. **Why discretize?** A table cannot enumerate infinitely many continuous states.
   Bins create a finite approximation, trading precision for table size.
5. **Why DQN?** A neural network generalizes Q-values across nearby continuous
   states. Replay and a target network reduce correlated and moving-target updates.
6. **How is PPO different?** DQN is an off-policy value method. PPO is an on-policy
   policy-gradient method that directly optimizes a stochastic policy while
   clipping large updates and learning a critic as a variance-reducing baseline.
7. **What do results mean?** Return and episode length both indicate how quickly
   the goal is reached; success rate captures reliability; multiple seeds and
   variance show stability; interaction count measures sample efficiency.

## Result to present

At 100,000 equal environment interactions over five final seeds, PPO was the only
reliable solver: final held-out return IQM **-81.7** and success IQM **100%**. DQN
was seed-sensitive (final return IQM **-434.7**, success **24%**) and sometimes
regressed after solving. Q-Learning was fastest but its discretized representation
did not reliably solve within the budget (final return IQM **-480.5**).

This is a conditional result for the tested environment, search spaces, seeds, and
budget—not a claim that PPO always beats value-based methods.

## Likely questions

- A true termination removes the future-value term; a time-limit truncation does not.
- Training exploration is disabled during evaluation.
- Equal episodes are not necessarily fair because episode lengths vary. This project
  budgets equal environment interactions and applies the same evaluator.
- PPO's clipped objective limits destructive policy changes, but does not guarantee
  that PPO will outperform DQN or Q-Learning on every budget.
- Final conclusions must be based on the raw CSV experiment output, not expectation.

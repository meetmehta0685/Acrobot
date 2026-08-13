from config import CommonConfig, DQNConfig, PPOConfig, QLearningConfig
from training.train_dqn import train_dqn
from training.train_ppo import train_ppo
from training.train_q_learning import train_q_learning


def test_short_training_runs_for_all_algorithms() -> None:
    common = CommonConfig(seed=7, interaction_budget=40, max_episode_steps=20, eval_episodes=1)
    q_agent, q_records = train_q_learning(common, QLearningConfig(bins=(2,) * 6))
    dqn_agent, dqn_records = train_dqn(common, DQNConfig(
        hidden_sizes=(8,), replay_capacity=100, batch_size=4, learning_starts=4,
        train_frequency=1, target_sync_frequency=5,
    ))
    ppo_agent, ppo_records = train_ppo(common, PPOConfig(
        hidden_sizes=(8,), rollout_steps=16, optimization_epochs=1, minibatch_size=8,
    ))
    assert len(q_records) == len(dqn_records) == len(ppo_records) == 2


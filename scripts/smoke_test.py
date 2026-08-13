"""Verify that Gymnasium's Acrobot environment is installed and usable."""

import gymnasium as gym


def main() -> None:
    env = gym.make("Acrobot-v1")
    observation, info = env.reset(seed=42)
    env.action_space.seed(42)
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print(f"Observation shape: {observation.shape}")
    print(f"Number of actions: {env.action_space.n}")
    total_reward = 0.0
    for _ in range(10):
        observation, reward, terminated, truncated, info = env.step(env.action_space.sample())
        total_reward += reward
        if terminated or truncated:
            observation, info = env.reset()
    env.close()
    print(f"10 random actions completed; reward={total_reward:.1f}")


if __name__ == "__main__":
    main()


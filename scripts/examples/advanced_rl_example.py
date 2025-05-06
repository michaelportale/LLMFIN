"""
Example script demonstrating the use of advanced reinforcement learning features.
"""

import os
import numpy as np
import gym
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

# Import our advanced RL modules
from scripts.advanced_rl import (
    HierarchicalRL,
    MultiAgentMarketSystem,
    ImitationLearning,
    CurriculumLearning,
    RuleBasedAgent
)

# Create output directory for results
os.makedirs("results/advanced_rl", exist_ok=True)


def create_simple_market_env(difficulty=0.5, noise_level=0.01, **kwargs):
    """
    Create a simple market environment with configurable difficulty.
    This is a placeholder - in a real implementation, you would create a proper market environment.
    """
    # For this example, we'll use CartPole as a stand-in
    env = gym.make('CartPole-v1')
    return env


def generate_expert_data(env, num_samples=1000):
    """
    Generate synthetic expert data for imitation learning.
    In a real application, this would be actual expert trader data.
    """
    # Train a PPO agent to generate "expert" data
    model = PPO("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=10000)
    
    # Generate expert demonstrations
    observations = []
    actions = []
    
    obs = env.reset()
    for _ in range(num_samples):
        action, _ = model.predict(obs, deterministic=True)
        observations.append(obs)
        actions.append(action)
        
        obs, _, done, _ = env.step(action)
        if done:
            obs = env.reset()
    
    return np.array(observations), np.array(actions)


def hierarchical_rl_example():
    """Example of hierarchical reinforcement learning."""
    print("\n=== Hierarchical Reinforcement Learning Example ===")
    
    # Create environment
    env = create_simple_market_env()
    
    # Create hierarchical RL agent
    hrl_agent = HierarchicalRL(
        env=env,
        num_options=3,
        meta_model_type="PPO",
        option_model_type="PPO",
        option_termination_proba=0.1,
        option_duration=5
    )
    
    # Train for a small number of steps (for demonstration)
    print("Training hierarchical RL agent...")
    hrl_agent.train(total_timesteps=10000)
    
    # Evaluate
    print("Evaluating hierarchical RL agent...")
    hrl_agent.evaluate(num_episodes=5)
    
    # Plot option usage
    hrl_agent.plot_option_usage(save_path="results/advanced_rl/hrl_option_usage.png")
    hrl_agent.plot_learning_curve(save_path="results/advanced_rl/hrl_learning_curve.png")
    
    # Save the agent
    hrl_agent.save("results/advanced_rl/hrl_agent")
    
    print("Hierarchical RL example completed!")


def multi_agent_example():
    """Example of multi-agent market simulation."""
    print("\n=== Multi-Agent Market Simulation Example ===")
    
    # Create environment
    env = create_simple_market_env()
    
    # Create multi-agent system
    market_system = MultiAgentMarketSystem(
        base_env=env,
        num_agents=3,
        agent_types=["PPO", "A2C", "rule_based"],
        agent_params=[
            {"learning_rate": 0.0003},
            {"learning_rate": 0.0007},
            {"strategy": "momentum", "threshold": 0.01}
        ],
        competitive_rewards=True,
        market_impact=True
    )
    
    # Train for a small number of episodes (for demonstration)
    print("Training multi-agent market system...")
    market_system.train(num_episodes=20, steps_per_episode=100)
    
    # Evaluate
    print("Evaluating multi-agent market system...")
    market_system.evaluate(num_episodes=5)
    
    # Plot rewards
    market_system.plot_rewards(save_path="results/advanced_rl/multi_agent_rewards.png")
    market_system.plot_market_impact(save_path="results/advanced_rl/market_impact.png")
    
    # Save the system
    market_system.save("results/advanced_rl/multi_agent_system")
    
    print("Multi-agent market simulation example completed!")


def imitation_learning_example():
    """Example of imitation learning from expert traders."""
    print("\n=== Imitation Learning Example ===")
    
    # Create environment
    env = create_simple_market_env()
    
    # Generate synthetic expert data (in a real application, load from file)
    print("Generating expert data...")
    expert_observations, expert_actions = generate_expert_data(env)
    
    # Create imitation learning agent
    il_agent = ImitationLearning(
        env=env,
        expert_data=(expert_observations, expert_actions),
        method='behavioral_cloning',
        model_type='mlp',
        hidden_sizes=[64, 32]
    )
    
    # Train
    print("Training imitation learning agent...")
    il_agent.train(epochs=10, batch_size=32, validation_split=0.2)
    
    # Evaluate
    print("Evaluating imitation learning agent...")
    il_agent.evaluate(num_episodes=5)
    
    # Plot learning curve
    il_agent.plot_learning_curve(save_path="results/advanced_rl/imitation_learning_curve.png")
    
    # Save the agent
    il_agent.save("results/advanced_rl/il_agent")
    
    print("Imitation learning example completed!")


def curriculum_learning_example():
    """Example of curriculum learning for progressive training difficulty."""
    print("\n=== Curriculum Learning Example ===")
    
    # Create curriculum learning system
    cl_system = CurriculumLearning(
        base_env_fn=create_simple_market_env,
        agent_type="PPO",
        num_stages=3,
        stage_durations=[2000, 3000, 5000],
        auto_progression=True,
        progression_threshold=0.7,
        stage_params=[
            {"difficulty": 0.2, "noise_level": 0.01},
            {"difficulty": 0.5, "noise_level": 0.02},
            {"difficulty": 0.8, "noise_level": 0.03}
        ]
    )
    
    # Train
    print("Training with curriculum learning...")
    cl_system.train(eval_freq=1000)
    
    # Plot learning curves
    cl_system.plot_learning_curve(save_path="results/advanced_rl/curriculum_learning_curve.png")
    cl_system.plot_stage_rewards(save_path="results/advanced_rl/curriculum_stage_rewards.png")
    
    # Save the system
    cl_system.save("results/advanced_rl/curriculum_system")
    
    print("Curriculum learning example completed!")


if __name__ == "__main__":
    print("Running advanced reinforcement learning examples...")
    
    # Run examples
    hierarchical_rl_example()
    multi_agent_example()
    imitation_learning_example()
    curriculum_learning_example()
    
    print("\nAll examples completed successfully!")
    print("Results saved to the 'results/advanced_rl/' directory") 
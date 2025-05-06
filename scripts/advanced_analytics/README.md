# Advanced Reinforcement Learning Features

This module provides advanced reinforcement learning capabilities for financial market modeling and trading strategies.

## Features

### 1. Hierarchical Reinforcement Learning

Hierarchical RL implements a two-level decision-making process:
- A meta-controller selects high-level actions (options)
- Each option is a sub-policy that executes for multiple timesteps
- This approach helps with long-term planning and exploration

```python
from scripts.advanced_rl import HierarchicalRL

# Create hierarchical RL agent
hrl_agent = HierarchicalRL(
    env=trading_env,
    num_options=3,
    meta_model_type="PPO",
    option_model_type="PPO"
)

# Train the agent
hrl_agent.train(total_timesteps=100000)

# Evaluate performance
hrl_agent.evaluate(num_episodes=10)

# Visualize option usage
hrl_agent.plot_option_usage()
```

### 2. Multi-Agent Systems for Market Simulation

Multi-agent systems allow realistic market simulations with multiple competing strategies:
- Multiple agents interact in a shared market environment
- Agents can have different strategies and learning algorithms
- Market impact is modeled based on agent actions

```python
from scripts.advanced_rl import MultiAgentMarketSystem, RuleBasedAgent

# Create multi-agent market system
market_system = MultiAgentMarketSystem(
    base_env=market_env,
    num_agents=5,
    agent_types=["PPO", "A2C", "SAC", "rule_based", "rule_based"],
    agent_params=[
        {"learning_rate": 0.0003},
        {"learning_rate": 0.0007},
        {"learning_rate": 0.0003},
        {"strategy": "momentum"},
        {"strategy": "mean_reversion"}
    ],
    competitive_rewards=True,
    market_impact=True
)

# Train the system
market_system.train(num_episodes=500)

# Evaluate performance
market_system.evaluate(num_episodes=20)

# Visualize agent rewards
market_system.plot_rewards()
```

### 3. Imitation Learning from Expert Traders

Learn trading strategies by mimicking expert behavior:
- Behavioral cloning directly learns a policy from expert demonstrations
- Inverse reinforcement learning infers the reward function from expert behavior
- Helps bootstrap RL agents with human expertise

```python
from scripts.advanced_rl import ImitationLearning

# Create imitation learning agent with expert data
il_agent = ImitationLearning(
    env=trading_env,
    expert_data_path="data/expert_trades.csv",
    method='behavioral_cloning',
    model_type='lstm',
    hidden_sizes=[128, 64]
)

# Train the agent
il_agent.train(epochs=100, batch_size=64)

# Evaluate performance
il_agent.evaluate(num_episodes=20)

# Visualize learning progress
il_agent.plot_learning_curve()
```

### 4. Curriculum Learning for Progressive Training

Gradually increase training difficulty to improve learning efficiency:
- Start with simple market scenarios and progressively increase complexity
- Automatically advance stages based on agent performance
- Helps agents learn complex strategies more efficiently

```python
from scripts.advanced_rl import CurriculumLearning

# Define environment creation function with difficulty parameters
def create_market_env(difficulty=0.5, noise_level=0.01, **kwargs):
    return MarketEnv(
        volatility=0.1 + difficulty * 0.2,
        noise_level=noise_level,
        **kwargs
    )

# Create curriculum learning system
curriculum = CurriculumLearning(
    base_env_fn=create_market_env,
    agent_type="PPO",
    num_stages=5,
    auto_progression=True,
    progression_threshold=0.7
)

# Train with curriculum
curriculum.train(total_timesteps=1000000)

# Visualize learning across stages
curriculum.plot_learning_curve()
curriculum.plot_stage_rewards()
```

## Integration with Existing Systems

These advanced RL features can be integrated with the existing trading system:

1. Use hierarchical RL for multi-timeframe trading strategies
2. Create realistic market simulations with multi-agent systems
3. Bootstrap RL agents with expert trader data using imitation learning
4. Train agents more efficiently with curriculum learning

## Requirements

- TensorFlow 2.x
- PyTorch
- Stable-Baselines3
- Gym
- NumPy
- Pandas
- Matplotlib 
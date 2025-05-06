import numpy as np
import tensorflow as tf
import gym
from gym import spaces
from stable_baselines3 import PPO, A2C, SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
import pandas as pd
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import json
import time
from typing import Dict, List, Tuple, Callable, Optional, Union, Any

class HierarchicalRL:
    """
    Hierarchical Reinforcement Learning implementation that uses a meta-controller
    to select sub-policies (options) that operate at different time scales.
    
    This implementation follows the Options framework where:
    - A meta-controller selects high-level actions (options)
    - Each option is a sub-policy that executes for multiple timesteps
    - Options have their own termination conditions
    """
    
    def __init__(
        self,
        env,
        num_options: int = 3,
        meta_model_type: str = "PPO",
        option_model_type: str = "PPO",
        meta_model_kwargs: Dict = None,
        option_model_kwargs: Dict = None,
        option_termination_proba: float = 0.1,
        option_duration: int = 10
    ):
        """
        Initialize the hierarchical RL agent.
        
        Args:
            env: Training environment
            num_options: Number of options (sub-policies)
            meta_model_type: Type of model for meta-controller
            option_model_type: Type of model for options
            meta_model_kwargs: Keyword arguments for meta-controller model
            option_model_kwargs: Keyword arguments for option models
            option_termination_proba: Probability of option termination
            option_duration: Maximum duration of an option
        """
        self.env = env
        self.num_options = num_options
        self.meta_model_type = meta_model_type
        self.option_model_type = option_model_type
        self.meta_model_kwargs = meta_model_kwargs or {}
        self.option_model_kwargs = option_model_kwargs or {}
        self.option_termination_proba = option_termination_proba
        self.option_duration = option_duration
        
        # Create meta-controller environment wrapper
        self.meta_env = self._create_meta_env()
        
        # Initialize models
        self.meta_controller = None
        self.options = []
        
        # Training history
        self.history = {
            'episode_rewards': [],
            'option_usage': {i: 0 for i in range(num_options)},
            'option_duration': {i: [] for i in range(num_options)}
        }
        
    def _create_meta_env(self):
        """
        Create a wrapper environment for the meta-controller.
        The meta-controller selects options and gets rewards accumulated over option execution.
        """
        # Create a custom meta-environment that wraps the original environment
        class MetaEnv(gym.Wrapper):
            def __init__(self, env, num_options):
                super().__init__(env)
                # Meta-controller selects from discrete options
                self.action_space = spaces.Discrete(num_options)
                # Observation space remains the same as the base environment
                
            def step(self, action):
                # This will be implemented by the execute_option method
                # This is just a placeholder
                return self.observation_space.sample(), 0, False, {}
                
            def reset(self):
                return self.env.reset()
        
        return MetaEnv(self.env, self.num_options)
    
    def _create_model(self, model_type, env, **kwargs):
        """Create a model based on the specified type."""
        if model_type == "PPO":
            return PPO("MlpPolicy", env, verbose=0, **kwargs)
        elif model_type == "A2C":
            return A2C("MlpPolicy", env, verbose=0, **kwargs)
        elif model_type == "SAC":
            return SAC("MlpPolicy", env, verbose=0, **kwargs)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def initialize_models(self):
        """Initialize the meta-controller and option models."""
        # Create meta-controller
        self.meta_controller = self._create_model(
            self.meta_model_type,
            self.meta_env,
            **self.meta_model_kwargs
        )
        
        # Create option models
        self.options = [
            self._create_model(
                self.option_model_type,
                self.env,
                **self.option_model_kwargs
            )
            for _ in range(self.num_options)
        ]
    
    def execute_option(self, option_idx, obs):
        """
        Execute the selected option until termination condition is met.
        
        Args:
            option_idx: Index of the option to execute
            obs: Current observation
            
        Returns:
            next_obs: Next observation after option execution
            total_reward: Accumulated reward during option execution
            done: Whether the episode has ended
            steps: Number of steps the option ran for
            info: Additional information
        """
        option = self.options[option_idx]
        total_reward = 0
        done = False
        info = {}
        steps = 0
        
        # Execute option for up to option_duration steps
        for _ in range(self.option_duration):
            # Select action using the option policy
            action, _ = option.predict(obs, deterministic=False)
            
            # Execute action in environment
            next_obs, reward, done, info = self.env.step(action)
            
            # Accumulate reward
            total_reward += reward
            steps += 1
            
            # Update observation
            obs = next_obs
            
            # Check termination conditions
            if done:
                break
                
            # Probabilistic termination of option
            if random.random() < self.option_termination_proba:
                break
        
        # Track option usage and duration
        self.history['option_usage'][option_idx] += 1
        self.history['option_duration'][option_idx].append(steps)
        
        return next_obs, total_reward, done, steps, info
    
    def train(self, total_timesteps=100000, meta_train_freq=5):
        """
        Train the hierarchical RL agent.
        
        Args:
            total_timesteps: Total number of timesteps to train for
            meta_train_freq: How often to train the meta-controller (in episodes)
        """
        if self.meta_controller is None or not self.options:
            self.initialize_models()
            
        timesteps = 0
        episode_count = 0
        option_buffer = []
        
        while timesteps < total_timesteps:
            obs = self.env.reset()
            done = False
            episode_reward = 0
            
            # Episode loop
            while not done and timesteps < total_timesteps:
                # Meta-controller selects option
                option_idx, _ = self.meta_controller.predict(obs, deterministic=False)
                option_start_obs = obs
                
                # Execute option
                next_obs, option_reward, done, option_steps, info = self.execute_option(option_idx, obs)
                
                # Store experience for meta-controller training
                option_buffer.append({
                    'obs': option_start_obs,
                    'option': option_idx,
                    'reward': option_reward,
                    'next_obs': next_obs,
                    'done': done
                })
                
                # Update state and rewards
                obs = next_obs
                episode_reward += option_reward
                timesteps += option_steps
                
                # Train option models every step
                self.options[option_idx].learn(total_timesteps=1, reset_num_timesteps=False)
            
            # End of episode
            episode_count += 1
            self.history['episode_rewards'].append(episode_reward)
            
            # Train meta-controller periodically
            if episode_count % meta_train_freq == 0 and option_buffer:
                self._train_meta_controller(option_buffer)
                option_buffer = []
                
            if episode_count % 10 == 0:
                mean_reward = np.mean(self.history['episode_rewards'][-10:])
                print(f"Episode {episode_count}, Mean Reward: {mean_reward:.2f}, Timesteps: {timesteps}/{total_timesteps}")
    
    def _train_meta_controller(self, option_buffer):
        """
        Train the meta-controller using collected experiences.
        
        Args:
            option_buffer: Buffer of option-level experiences
        """
        # Create a custom environment for the meta-controller to learn from the buffer
        class BufferEnv(gym.Env):
            def __init__(self, buffer, observation_space, action_space):
                self.buffer = buffer
                self.current_idx = 0
                self.observation_space = observation_space
                self.action_space = action_space
                
            def reset(self):
                self.current_idx = random.randint(0, len(self.buffer) - 1)
                return self.buffer[self.current_idx]['obs']
                
            def step(self, action):
                experience = self.buffer[self.current_idx]
                reward = experience['reward'] if action == experience['option'] else 0
                next_obs = experience['next_obs']
                done = experience['done']
                
                # Move to next experience
                self.current_idx = (self.current_idx + 1) % len(self.buffer)
                
                return next_obs, reward, done, {}
        
        buffer_env = DummyVecEnv([lambda: BufferEnv(
            option_buffer, 
            self.env.observation_space, 
            spaces.Discrete(self.num_options)
        )])
        
        # Train meta-controller on buffer
        steps = min(len(option_buffer) * 10, 1000)  # Adjust based on buffer size
        self.meta_controller.set_env(buffer_env)
        self.meta_controller.learn(total_timesteps=steps, reset_num_timesteps=False)
        self.meta_controller.set_env(self.meta_env)
    
    def evaluate(self, num_episodes=10):
        """
        Evaluate the hierarchical RL agent.
        
        Args:
            num_episodes: Number of episodes to evaluate
            
        Returns:
            mean_reward: Mean reward across episodes
        """
        if self.meta_controller is None or not self.options:
            raise ValueError("Agent must be trained before evaluation")
            
        rewards = []
        
        for _ in range(num_episodes):
            obs = self.env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                # Meta-controller selects option
                option_idx, _ = self.meta_controller.predict(obs, deterministic=True)
                
                # Execute option
                next_obs, option_reward, done, _, _ = self.execute_option(option_idx, obs)
                
                # Update state and rewards
                obs = next_obs
                episode_reward += option_reward
            
            rewards.append(episode_reward)
        
        mean_reward = np.mean(rewards)
        print(f"Evaluation over {num_episodes} episodes: Mean Reward: {mean_reward:.2f}")
        return mean_reward
    
    def plot_option_usage(self, save_path=None):
        """
        Plot the usage distribution of options.
        
        Args:
            save_path: Path to save the plot
        """
        option_usage = self.history['option_usage']
        
        plt.figure(figsize=(10, 6))
        plt.bar(option_usage.keys(), option_usage.values())
        plt.xlabel('Option Index')
        plt.ylabel('Usage Count')
        plt.title('Option Usage Distribution')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    
    def plot_learning_curve(self, window=10, save_path=None):
        """
        Plot the learning curve (episode rewards).
        
        Args:
            window: Window size for smoothing
            save_path: Path to save the plot
        """
        rewards = self.history['episode_rewards']
        
        # Compute rolling average
        if len(rewards) >= window:
            smoothed_rewards = [np.mean(rewards[i:i+window]) for i in range(len(rewards) - window + 1)]
            episodes = list(range(window-1, len(rewards)))
        else:
            smoothed_rewards = rewards
            episodes = list(range(len(rewards)))
        
        plt.figure(figsize=(10, 6))
        plt.plot(episodes, smoothed_rewards)
        plt.xlabel('Episode')
        plt.ylabel(f'Average Reward (Window {window})')
        plt.title('Learning Curve')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    
    def save(self, directory):
        """
        Save the hierarchical RL agent.
        
        Args:
            directory: Directory to save the agent
        """
        os.makedirs(directory, exist_ok=True)
        
        # Save meta-controller
        self.meta_controller.save(os.path.join(directory, "meta_controller"))
        
        # Save options
        for i, option in enumerate(self.options):
            option.save(os.path.join(directory, f"option_{i}"))
        
        # Save history and parameters
        with open(os.path.join(directory, "history.json"), "w") as f:
            # Convert numpy arrays to lists for JSON serialization
            history_json = {
                'episode_rewards': self.history['episode_rewards'],
                'option_usage': self.history['option_usage'],
                'option_duration': {k: v for k, v in self.history['option_duration'].items()}
            }
            json.dump(history_json, f)
        
        # Save parameters
        params = {
            'num_options': self.num_options,
            'meta_model_type': self.meta_model_type,
            'option_model_type': self.option_model_type,
            'option_termination_proba': self.option_termination_proba,
            'option_duration': self.option_duration
        }
        with open(os.path.join(directory, "params.json"), "w") as f:
            json.dump(params, f)
    
    @classmethod
    def load(cls, directory, env):
        """
        Load a hierarchical RL agent.
        
        Args:
            directory: Directory to load the agent from
            env: Environment to use
            
        Returns:
            agent: Loaded hierarchical RL agent
        """
        # Load parameters
        with open(os.path.join(directory, "params.json"), "r") as f:
            params = json.load(f)
        
        # Create agent with loaded parameters
        agent = cls(
            env=env,
            num_options=params['num_options'],
            meta_model_type=params['meta_model_type'],
            option_model_type=params['option_model_type'],
            option_termination_proba=params['option_termination_proba'],
            option_duration=params['option_duration']
        )
        
        # Initialize models
        agent.initialize_models()
        
        # Load meta-controller
        agent.meta_controller = agent.meta_controller.load(
            os.path.join(directory, "meta_controller"), 
            env=agent.meta_env
        )
        
        # Load options
        for i in range(agent.num_options):
            agent.options[i] = agent.options[i].load(
                os.path.join(directory, f"option_{i}"),
                env=env
            )
        
        # Load history
        with open(os.path.join(directory, "history.json"), "r") as f:
            agent.history = json.load(f)
        
        return agent 

class MultiAgentMarketSystem:
    """
    Multi-agent system for market simulation where multiple agents interact
    in a shared market environment. This enables the simulation of realistic
    market dynamics with competing strategies.
    """
    
    def __init__(
        self,
        base_env,
        num_agents: int = 5,
        agent_types: List[str] = None,
        agent_params: List[Dict] = None,
        competitive_rewards: bool = True,
        shared_observations: bool = False,
        market_impact: bool = True
    ):
        """
        Initialize the multi-agent market system.
        
        Args:
            base_env: Base market environment
            num_agents: Number of agents in the system
            agent_types: Types of agents (e.g., "PPO", "A2C", "SAC", "rule_based")
            agent_params: Parameters for each agent
            competitive_rewards: Whether agents compete for rewards
            shared_observations: Whether agents share observations
            market_impact: Whether agents' actions impact the market
        """
        self.base_env = base_env
        self.num_agents = num_agents
        
        # Default to all PPO agents if not specified
        self.agent_types = agent_types or ["PPO"] * num_agents
        self.agent_params = agent_params or [{} for _ in range(num_agents)]
        
        self.competitive_rewards = competitive_rewards
        self.shared_observations = shared_observations
        self.market_impact = market_impact
        
        # Initialize agents
        self.agents = []
        self._initialize_agents()
        
        # Tracking metrics
        self.history = {
            'episode_rewards': {i: [] for i in range(num_agents)},
            'market_states': [],
            'agent_actions': {i: [] for i in range(num_agents)},
            'market_impacts': []
        }
    
    def _initialize_agents(self):
        """Initialize all agents in the system."""
        for i in range(self.num_agents):
            agent_type = self.agent_types[i]
            params = self.agent_params[i]
            
            if agent_type in ["PPO", "A2C", "SAC"]:
                # Create RL agent
                if agent_type == "PPO":
                    agent = PPO("MlpPolicy", self.base_env, verbose=0, **params)
                elif agent_type == "A2C":
                    agent = A2C("MlpPolicy", self.base_env, verbose=0, **params)
                elif agent_type == "SAC":
                    agent = SAC("MlpPolicy", self.base_env, verbose=0, **params)
            elif agent_type == "rule_based":
                # Create rule-based agent
                agent = RuleBasedAgent(self.base_env, **params)
            else:
                raise ValueError(f"Unsupported agent type: {agent_type}")
                
            self.agents.append(agent)
    
    def _calculate_market_impact(self, actions):
        """
        Calculate the impact of all agents' actions on the market.
        
        Args:
            actions: List of actions from all agents
            
        Returns:
            market_impact: Impact on the market
        """
        if not self.market_impact:
            return 0
            
        # Simple model: impact proportional to sum of absolute actions
        return np.sum(np.abs(actions)) * 0.01  # Scale factor can be adjusted
    
    def _adjust_rewards(self, base_rewards, actions, market_impact):
        """
        Adjust rewards based on competition and market impact.
        
        Args:
            base_rewards: Base rewards from environment
            actions: List of actions from all agents
            market_impact: Calculated market impact
            
        Returns:
            adjusted_rewards: List of adjusted rewards for each agent
        """
        adjusted_rewards = base_rewards.copy()
        
        if self.competitive_rewards:
            # Zero-sum adjustment: rewards are redistributed based on relative performance
            total_reward = sum(base_rewards)
            performance = np.array(base_rewards) / (total_reward + 1e-10)  # Avoid division by zero
            
            for i in range(self.num_agents):
                # Agents get reward based on their relative performance
                adjusted_rewards[i] = total_reward * performance[i]
        
        if self.market_impact:
            # Penalize all agents for market impact, proportional to their action magnitude
            action_magnitudes = np.abs(actions)
            total_magnitude = np.sum(action_magnitudes) + 1e-10  # Avoid division by zero
            
            for i in range(self.num_agents):
                # Penalty proportional to agent's contribution to market impact
                impact_penalty = market_impact * (action_magnitudes[i] / total_magnitude)
                adjusted_rewards[i] -= impact_penalty
        
        return adjusted_rewards
    
    def step(self, observations):
        """
        Execute one step with all agents.
        
        Args:
            observations: Current observations (list or single observation)
            
        Returns:
            next_observations: Next observations
            rewards: Rewards for each agent
            dones: Done flags for each agent
            infos: Info dictionaries for each agent
        """
        actions = []
        
        # Get actions from all agents
        for i, agent in enumerate(self.agents):
            if self.shared_observations:
                obs = observations  # All agents see the same observation
            else:
                obs = observations[i]  # Each agent has its own observation
                
            if hasattr(agent, 'predict'):
                # RL agent
                action, _ = agent.predict(obs, deterministic=False)
            else:
                # Rule-based agent
                action = agent.act(obs)
                
            actions.append(action)
        
        # Calculate market impact
        market_impact = self._calculate_market_impact(actions)
        
        # Execute actions in environment
        if self.market_impact:
            # Add market impact to the environment state
            next_obs, base_rewards, done, info = self.base_env.step(
                actions, market_impact=market_impact
            )
        else:
            # Standard environment step
            next_obs, base_rewards, done, info = self.base_env.step(actions)
        
        # Adjust rewards based on competition and market impact
        if isinstance(base_rewards, (int, float)):
            base_rewards = [base_rewards] * self.num_agents
            
        adjusted_rewards = self._adjust_rewards(base_rewards, actions, market_impact)
        
        # Update history
        for i in range(self.num_agents):
            self.history['agent_actions'][i].append(actions[i])
            
            if len(self.history['episode_rewards'][i]) == 0:
                self.history['episode_rewards'][i].append(adjusted_rewards[i])
            else:
                self.history['episode_rewards'][i][-1] += adjusted_rewards[i]
        
        self.history['market_impacts'].append(market_impact)
        
        return next_obs, adjusted_rewards, done, info
    
    def reset(self):
        """
        Reset the environment and start a new episode.
        
        Returns:
            observations: Initial observations
        """
        observations = self.base_env.reset()
        
        # Start new episode in history
        for i in range(self.num_agents):
            self.history['episode_rewards'][i].append(0)
            
        return observations
    
    def train(self, num_episodes=100, steps_per_episode=None):
        """
        Train all agents in the multi-agent system.
        
        Args:
            num_episodes: Number of episodes to train for
            steps_per_episode: Maximum steps per episode (None for no limit)
        """
        for episode in range(num_episodes):
            observations = self.reset()
            done = False
            step = 0
            
            while not done:
                # Check step limit
                if steps_per_episode and step >= steps_per_episode:
                    break
                    
                # Execute step
                next_observations, rewards, done, _ = self.step(observations)
                
                # Update observations
                observations = next_observations
                
                # Train all RL agents
                for i, agent in enumerate(self.agents):
                    if hasattr(agent, 'learn'):
                        # Only RL agents have learn method
                        agent.learn(total_timesteps=1, reset_num_timesteps=False)
                
                step += 1
            
            # Print episode summary
            if episode % 10 == 0:
                mean_rewards = [np.mean(self.history['episode_rewards'][i][-10:]) 
                               for i in range(self.num_agents)]
                print(f"Episode {episode}/{num_episodes}")
                for i, reward in enumerate(mean_rewards):
                    print(f"  Agent {i} ({self.agent_types[i]}): {reward:.2f}")
    
    def evaluate(self, num_episodes=10):
        """
        Evaluate all agents in the multi-agent system.
        
        Args:
            num_episodes: Number of episodes to evaluate
            
        Returns:
            mean_rewards: Mean rewards for each agent
        """
        eval_rewards = {i: [] for i in range(self.num_agents)}
        
        for _ in range(num_episodes):
            observations = self.reset()
            done = False
            
            while not done:
                # Get actions deterministically for evaluation
                actions = []
                for i, agent in enumerate(self.agents):
                    if hasattr(agent, 'predict'):
                        # RL agent
                        action, _ = agent.predict(observations, deterministic=True)
                    else:
                        # Rule-based agent
                        action = agent.act(observations)
                        
                    actions.append(action)
                
                # Execute actions
                observations, rewards, done, _ = self.step(observations)
                
                # Record rewards
                for i, reward in enumerate(rewards):
                    if len(eval_rewards[i]) < num_episodes:
                        eval_rewards[i].append(0)
                    eval_rewards[i][-1] += reward
        
        # Calculate mean rewards
        mean_rewards = {i: np.mean(rewards) for i, rewards in eval_rewards.items()}
        
        print("Evaluation results:")
        for i, reward in mean_rewards.items():
            print(f"  Agent {i} ({self.agent_types[i]}): {reward:.2f}")
            
        return mean_rewards
    
    def plot_rewards(self, window=10, save_path=None):
        """
        Plot the reward history for all agents.
        
        Args:
            window: Window size for smoothing
            save_path: Path to save the plot
        """
        plt.figure(figsize=(12, 8))
        
        for i in range(self.num_agents):
            rewards = self.history['episode_rewards'][i]
            
            # Compute rolling average
            if len(rewards) >= window:
                smoothed_rewards = [np.mean(rewards[j:j+window]) 
                                   for j in range(len(rewards) - window + 1)]
                episodes = list(range(window-1, len(rewards)))
            else:
                smoothed_rewards = rewards
                episodes = list(range(len(rewards)))
            
            plt.plot(episodes, smoothed_rewards, label=f"Agent {i} ({self.agent_types[i]})")
        
        plt.xlabel('Episode')
        plt.ylabel(f'Average Reward (Window {window})')
        plt.title('Multi-Agent Learning Curves')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    
    def plot_market_impact(self, window=10, save_path=None):
        """
        Plot the market impact over time.
        
        Args:
            window: Window size for smoothing
            save_path: Path to save the plot
        """
        impacts = self.history['market_impacts']
        
        # Compute rolling average
        if len(impacts) >= window:
            smoothed_impacts = [np.mean(impacts[i:i+window]) 
                               for i in range(len(impacts) - window + 1)]
            steps = list(range(window-1, len(impacts)))
        else:
            smoothed_impacts = impacts
            steps = list(range(len(impacts)))
        
        plt.figure(figsize=(10, 6))
        plt.plot(steps, smoothed_impacts)
        plt.xlabel('Step')
        plt.ylabel(f'Market Impact (Window {window})')
        plt.title('Market Impact Over Time')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    
    def save(self, directory):
        """
        Save the multi-agent system.
        
        Args:
            directory: Directory to save the system
        """
        os.makedirs(directory, exist_ok=True)
        
        # Save agents
        for i, agent in enumerate(self.agents):
            if hasattr(agent, 'save'):
                agent.save(os.path.join(directory, f"agent_{i}"))
        
        # Save history and parameters
        with open(os.path.join(directory, "history.json"), "w") as f:
            # Convert numpy arrays to lists for JSON serialization
            history_json = {
                'episode_rewards': self.history['episode_rewards'],
                'market_impacts': self.history['market_impacts']
            }
            json.dump(history_json, f)
        
        # Save parameters
        params = {
            'num_agents': self.num_agents,
            'agent_types': self.agent_types,
            'competitive_rewards': self.competitive_rewards,
            'shared_observations': self.shared_observations,
            'market_impact': self.market_impact
        }
        with open(os.path.join(directory, "params.json"), "w") as f:
            json.dump(params, f)


class RuleBasedAgent:
    """
    Simple rule-based agent for use in multi-agent systems.
    Can be used as a baseline or to create specific market behaviors.
    """
    
    def __init__(self, env, strategy='momentum', threshold=0.01, window=10):
        """
        Initialize the rule-based agent.
        
        Args:
            env: Environment the agent operates in
            strategy: Trading strategy ('momentum', 'mean_reversion', 'random')
            threshold: Action threshold
            window: Lookback window for strategies
        """
        self.env = env
        self.strategy = strategy
        self.threshold = threshold
        self.window = window
        self.price_history = deque(maxlen=window)
    
    def act(self, observation):
        """
        Select an action based on the rule-based strategy.
        
        Args:
            observation: Current observation
            
        Returns:
            action: Selected action
        """
        # Extract price from observation (assuming it's part of the observation)
        if hasattr(observation, 'shape') and len(observation.shape) > 0:
            # Assume price is the first feature
            price = observation[0]
        else:
            # Fallback
            price = observation
            
        # Store price in history
        self.price_history.append(price)
        
        # Not enough history yet
        if len(self.price_history) < 2:
            return 0  # No action (hold)
        
        if self.strategy == 'momentum':
            # Momentum strategy: follow the trend
            price_change = (price - self.price_history[-2]) / self.price_history[-2]
            
            if price_change > self.threshold:
                return 1  # Buy
            elif price_change < -self.threshold:
                return -1  # Sell
            else:
                return 0  # Hold
                
        elif self.strategy == 'mean_reversion':
            # Mean reversion: bet against the trend
            if len(self.price_history) < self.window:
                return 0  # Not enough history
                
            mean_price = np.mean(self.price_history)
            price_deviation = (price - mean_price) / mean_price
            
            if price_deviation > self.threshold:
                return -1  # Sell (expect reversion down)
            elif price_deviation < -self.threshold:
                return 1  # Buy (expect reversion up)
            else:
                return 0  # Hold
                
        elif self.strategy == 'random':
            # Random strategy
            return random.choice([-1, 0, 1])
            
        else:
            raise ValueError(f"Unsupported strategy: {self.strategy}") 

class ImitationLearning:
    """
    Imitation Learning implementation for learning trading strategies from expert traders.
    Supports behavioral cloning and inverse reinforcement learning approaches.
    """
    
    def __init__(
        self,
        env,
        expert_data=None,
        expert_data_path=None,
        method='behavioral_cloning',
        model_type='mlp',
        hidden_sizes=[64, 64],
        learning_rate=1e-4
    ):
        """
        Initialize the imitation learning system.
        
        Args:
            env: Training environment
            expert_data: Expert demonstration data (if already loaded)
            expert_data_path: Path to expert data file
            method: Imitation learning method ('behavioral_cloning' or 'inverse_rl')
            model_type: Type of model architecture ('mlp' or 'lstm')
            hidden_sizes: Hidden layer sizes for neural networks
            learning_rate: Learning rate for training
        """
        self.env = env
        self.method = method
        self.model_type = model_type
        self.hidden_sizes = hidden_sizes
        self.learning_rate = learning_rate
        
        # Load expert data
        if expert_data is not None:
            self.expert_data = expert_data
        elif expert_data_path is not None:
            self.expert_data = self._load_expert_data(expert_data_path)
        else:
            self.expert_data = None
            
        # Initialize models
        self.policy_network = None
        self.reward_network = None
        
        # Training history
        self.history = {
            'loss': [],
            'validation_accuracy': [],
            'episode_rewards': []
        }
    
    def _load_expert_data(self, path):
        """
        Load expert demonstration data from file.
        
        Args:
            path: Path to expert data file
            
        Returns:
            expert_data: Loaded expert data
        """
        # Determine file format and load accordingly
        if path.endswith('.csv'):
            return pd.read_csv(path)
        elif path.endswith('.npy'):
            return np.load(path, allow_pickle=True)
        elif path.endswith('.json'):
            with open(path, 'r') as f:
                return json.load(f)
        else:
            raise ValueError(f"Unsupported file format: {path}")
    
    def _preprocess_expert_data(self):
        """
        Preprocess expert data for training.
        
        Returns:
            observations: Expert observations
            actions: Expert actions
        """
        if isinstance(self.expert_data, pd.DataFrame):
            # Assume DataFrame with 'observation' and 'action' columns
            observations = np.array(self.expert_data['observation'].tolist())
            actions = np.array(self.expert_data['action'].tolist())
        elif isinstance(self.expert_data, dict):
            # Assume dictionary with 'observations' and 'actions' keys
            observations = np.array(self.expert_data['observations'])
            actions = np.array(self.expert_data['actions'])
        elif isinstance(self.expert_data, tuple) and len(self.expert_data) == 2:
            # Assume tuple of (observations, actions)
            observations, actions = self.expert_data
        else:
            raise ValueError("Expert data format not recognized")
            
        return observations, actions
    
    def _create_policy_network(self):
        """
        Create the policy network for behavioral cloning.
        
        Returns:
            model: Created policy network
        """
        if not tf.executing_eagerly():
            tf.compat.v1.enable_eager_execution()
        
        # Get input shape from environment
        input_shape = self.env.observation_space.shape
        
        # Get output shape from environment
        if isinstance(self.env.action_space, spaces.Discrete):
            output_dim = self.env.action_space.n  # Discrete actions
            activation = 'softmax'
            loss = 'sparse_categorical_crossentropy'
        else:
            output_dim = self.env.action_space.shape[0]  # Continuous actions
            activation = 'tanh'  # Assuming actions are in [-1, 1]
            loss = 'mse'
        
        # Create model
        model = tf.keras.Sequential()
        
        # Input layer
        model.add(tf.keras.layers.InputLayer(input_shape=input_shape))
        
        if self.model_type == 'mlp':
            # MLP architecture
            for hidden_size in self.hidden_sizes:
                model.add(tf.keras.layers.Dense(hidden_size, activation='relu'))
                model.add(tf.keras.layers.BatchNormalization())
                model.add(tf.keras.layers.Dropout(0.2))
        elif self.model_type == 'lstm':
            # LSTM architecture
            # Reshape input for LSTM if needed
            if len(input_shape) == 1:
                model.add(tf.keras.layers.Reshape((1, input_shape[0])))
                
            for i, hidden_size in enumerate(self.hidden_sizes):
                return_sequences = i < len(self.hidden_sizes) - 1
                model.add(tf.keras.layers.LSTM(hidden_size, return_sequences=return_sequences))
                model.add(tf.keras.layers.Dropout(0.2))
        
        # Output layer
        model.add(tf.keras.layers.Dense(output_dim, activation=activation))
        
        # Compile model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss=loss,
            metrics=['accuracy']
        )
        
        return model
    
    def _create_reward_network(self):
        """
        Create the reward network for inverse RL.
        
        Returns:
            model: Created reward network
        """
        if not tf.executing_eagerly():
            tf.compat.v1.enable_eager_execution()
        
        # Get input shape from environment (state + action)
        state_shape = self.env.observation_space.shape[0]
        
        if isinstance(self.env.action_space, spaces.Discrete):
            action_shape = self.env.action_space.n  # One-hot encoding for discrete actions
        else:
            action_shape = self.env.action_space.shape[0]  # Continuous actions
            
        input_shape = (state_shape + action_shape,)
        
        # Create model
        model = tf.keras.Sequential()
        
        # Input layer
        model.add(tf.keras.layers.InputLayer(input_shape=input_shape))
        
        # Hidden layers
        for hidden_size in self.hidden_sizes:
            model.add(tf.keras.layers.Dense(hidden_size, activation='relu'))
            model.add(tf.keras.layers.BatchNormalization())
            model.add(tf.keras.layers.Dropout(0.2))
        
        # Output layer (scalar reward)
        model.add(tf.keras.layers.Dense(1))
        
        # Compile model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse'
        )
        
        return model
    
    def train_behavioral_cloning(self, validation_split=0.2, epochs=100, batch_size=64):
        """
        Train the policy network using behavioral cloning.
        
        Args:
            validation_split: Fraction of data to use for validation
            epochs: Number of training epochs
            batch_size: Training batch size
            
        Returns:
            history: Training history
        """
        if self.expert_data is None:
            raise ValueError("Expert data is required for behavioral cloning")
            
        # Preprocess expert data
        observations, actions = self._preprocess_expert_data()
        
        # Create policy network if not already created
        if self.policy_network is None:
            self.policy_network = self._create_policy_network()
            
        # Train the model
        history = self.policy_network.fit(
            observations, actions,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )
        
        # Store training history
        self.history['loss'].extend(history.history['loss'])
        self.history['validation_accuracy'].extend(history.history['val_accuracy'])
        
        return history
    
    def train_inverse_rl(self, num_iterations=100, rl_steps_per_iter=10000, batch_size=64):
        """
        Train using inverse reinforcement learning (IRL).
        This implements a basic version of the GAIL algorithm.
        
        Args:
            num_iterations: Number of IRL iterations
            rl_steps_per_iter: RL steps per iteration
            batch_size: Batch size for reward network training
            
        Returns:
            policy: Trained RL policy
        """
        if self.expert_data is None:
            raise ValueError("Expert data is required for inverse RL")
            
        # Preprocess expert data
        expert_observations, expert_actions = self._preprocess_expert_data()
        
        # Create reward network if not already created
        if self.reward_network is None:
            self.reward_network = self._create_reward_network()
            
        # Create RL policy
        policy = PPO("MlpPolicy", self.env, verbose=0)
        
        for iteration in range(num_iterations):
            print(f"IRL Iteration {iteration+1}/{num_iterations}")
            
            # Collect experience with current policy
            observations = []
            actions = []
            rewards = []
            
            obs = self.env.reset()
            for _ in range(1000):  # Collect 1000 steps
                action, _ = policy.predict(obs)
                next_obs, _, done, _ = self.env.step(action)
                
                observations.append(obs)
                actions.append(action)
                
                obs = next_obs
                if done:
                    obs = self.env.reset()
            
            # Convert to numpy arrays
            observations = np.array(observations)
            actions = np.array(actions)
            
            # Prepare discriminator data
            if isinstance(self.env.action_space, spaces.Discrete):
                # One-hot encode discrete actions
                policy_actions_oh = np.eye(self.env.action_space.n)[actions]
                expert_actions_oh = np.eye(self.env.action_space.n)[expert_actions]
                
                policy_inputs = np.concatenate([observations, policy_actions_oh], axis=1)
                expert_inputs = np.concatenate([expert_observations, expert_actions_oh], axis=1)
            else:
                policy_inputs = np.concatenate([observations, actions], axis=1)
                expert_inputs = np.concatenate([expert_observations, expert_actions], axis=1)
            
            # Labels: 0 for policy data, 1 for expert data
            policy_labels = np.zeros((len(observations), 1))
            expert_labels = np.ones((len(expert_observations), 1))
            
            # Train discriminator (reward network)
            combined_inputs = np.concatenate([policy_inputs, expert_inputs], axis=0)
            combined_labels = np.concatenate([policy_labels, expert_labels], axis=0)
            
            # Shuffle the data
            indices = np.random.permutation(len(combined_inputs))
            combined_inputs = combined_inputs[indices]
            combined_labels = combined_labels[indices]
            
            # Train discriminator
            disc_history = self.reward_network.fit(
                combined_inputs, combined_labels,
                epochs=5,
                batch_size=batch_size,
                verbose=0
            )
            
            # Use discriminator as reward function
            def reward_fn(obs, action):
                if isinstance(self.env.action_space, spaces.Discrete):
                    action_oh = np.eye(self.env.action_space.n)[action]
                    inputs = np.concatenate([obs.reshape(1, -1), action_oh.reshape(1, -1)], axis=1)
                else:
                    inputs = np.concatenate([obs.reshape(1, -1), action.reshape(1, -1)], axis=1)
                
                # Log probability as reward
                reward = -np.log(1 - self.reward_network.predict(inputs, verbose=0)[0][0] + 1e-10)
                return reward
            
            # Create a reward wrapper for the environment
            class RewardWrapper(gym.RewardWrapper):
                def __init__(self, env, reward_function):
                    super().__init__(env)
                    self.reward_function = reward_function
                    
                def reward(self, reward):
                    obs = self.unwrapped._get_obs()
                    action = self.unwrapped.last_action
                    return self.reward_function(obs, action)
            
            # Wrap environment with custom reward
            wrapped_env = RewardWrapper(self.env, reward_fn)
            policy.set_env(wrapped_env)
            
            # Update policy with RL
            policy.learn(total_timesteps=rl_steps_per_iter)
            
            # Evaluate policy
            eval_rewards = []
            for _ in range(5):  # 5 evaluation episodes
                obs = self.env.reset()
                done = False
                total_reward = 0
                
                while not done:
                    action, _ = policy.predict(obs, deterministic=True)
                    obs, reward, done, _ = self.env.step(action)
                    total_reward += reward
                
                eval_rewards.append(total_reward)
            
            mean_reward = np.mean(eval_rewards)
            self.history['episode_rewards'].append(mean_reward)
            
            print(f"  Discriminator loss: {disc_history.history['loss'][-1]:.4f}")
            print(f"  Policy evaluation: {mean_reward:.2f}")
        
        return policy
    
    def train(self, **kwargs):
        """
        Train the imitation learning model based on the selected method.
        
        Args:
            **kwargs: Additional arguments for the specific training method
            
        Returns:
            Model or history object
        """
        if self.method == 'behavioral_cloning':
            return self.train_behavioral_cloning(**kwargs)
        elif self.method == 'inverse_rl':
            return self.train_inverse_rl(**kwargs)
        else:
            raise ValueError(f"Unsupported method: {self.method}")
    
    def predict(self, observation):
        """
        Make a prediction using the trained policy.
        
        Args:
            observation: Current observation
            
        Returns:
            action: Predicted action
        """
        if self.method == 'behavioral_cloning':
            if self.policy_network is None:
                raise ValueError("Policy network must be trained before prediction")
                
            # Reshape observation if needed
            if len(observation.shape) == 1:
                observation = observation.reshape(1, -1)
                
            # Get prediction
            prediction = self.policy_network.predict(observation, verbose=0)
            
            # Handle discrete vs continuous actions
            if isinstance(self.env.action_space, spaces.Discrete):
                return np.argmax(prediction, axis=1)[0]
            else:
                return prediction[0]
        else:
            raise ValueError("Prediction is only supported for behavioral cloning")
    
    def evaluate(self, num_episodes=10):
        """
        Evaluate the trained policy.
        
        Args:
            num_episodes: Number of episodes to evaluate
            
        Returns:
            mean_reward: Mean reward across episodes
        """
        rewards = []
        
        for _ in range(num_episodes):
            obs = self.env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                action = self.predict(obs)
                obs, reward, done, _ = self.env.step(action)
                episode_reward += reward
            
            rewards.append(episode_reward)
        
        mean_reward = np.mean(rewards)
        print(f"Evaluation over {num_episodes} episodes: Mean Reward: {mean_reward:.2f}")
        return mean_reward
    
    def plot_learning_curve(self, save_path=None):
        """
        Plot the learning curve.
        
        Args:
            save_path: Path to save the plot
        """
        plt.figure(figsize=(12, 8))
        
        if self.method == 'behavioral_cloning':
            # Plot training and validation metrics
            plt.subplot(2, 1, 1)
            plt.plot(self.history['loss'], label='Training Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Behavioral Cloning Training Loss')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            plt.subplot(2, 1, 2)
            plt.plot(self.history['validation_accuracy'], label='Validation Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.title('Behavioral Cloning Validation Accuracy')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
        elif self.method == 'inverse_rl':
            # Plot episode rewards
            plt.plot(self.history['episode_rewards'])
            plt.xlabel('Iteration')
            plt.ylabel('Mean Reward')
            plt.title('Inverse RL Learning Curve')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    
    def save(self, directory):
        """
        Save the imitation learning model.
        
        Args:
            directory: Directory to save the model
        """
        os.makedirs(directory, exist_ok=True)
        
        # Save policy network
        if self.policy_network is not None:
            self.policy_network.save(os.path.join(directory, "policy_network"))
        
        # Save reward network
        if self.reward_network is not None:
            self.reward_network.save(os.path.join(directory, "reward_network"))
        
        # Save history and parameters
        with open(os.path.join(directory, "history.json"), "w") as f:
            json.dump(self.history, f)
        
        # Save parameters
        params = {
            'method': self.method,
            'model_type': self.model_type,
            'hidden_sizes': self.hidden_sizes,
            'learning_rate': self.learning_rate
        }
        with open(os.path.join(directory, "params.json"), "w") as f:
            json.dump(params, f)
    
    @classmethod
    def load(cls, directory, env, expert_data=None):
        """
        Load an imitation learning model.
        
        Args:
            directory: Directory to load the model from
            env: Environment to use
            expert_data: Expert data (optional)
            
        Returns:
            model: Loaded imitation learning model
        """
        # Load parameters
        with open(os.path.join(directory, "params.json"), "r") as f:
            params = json.load(f)
        
        # Create model with loaded parameters
        model = cls(
            env=env,
            expert_data=expert_data,
            method=params['method'],
            model_type=params['model_type'],
            hidden_sizes=params['hidden_sizes'],
            learning_rate=params['learning_rate']
        )
        
        # Load policy network if it exists
        policy_path = os.path.join(directory, "policy_network")
        if os.path.exists(policy_path):
            model.policy_network = tf.keras.models.load_model(policy_path)
        
        # Load reward network if it exists
        reward_path = os.path.join(directory, "reward_network")
        if os.path.exists(reward_path):
            model.reward_network = tf.keras.models.load_model(reward_path)
        
        # Load history
        with open(os.path.join(directory, "history.json"), "r") as f:
            model.history = json.load(f)
        
        return model 
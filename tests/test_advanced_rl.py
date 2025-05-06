import os
import sys
import unittest
import tempfile
import shutil
import gym
import numpy as np
import tensorflow as tf
from unittest.mock import MagicMock, patch

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.advanced_rl import HierarchicalRL, MultiAgentMarketSystem, ImitationLearning

class TestCartPoleEnv(gym.Env):
    """Simple test environment based on CartPole."""
    
    def __init__(self):
        super().__init__()
        self.env = gym.make('CartPole-v1')
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        
    def reset(self):
        return self.env.reset()
    
    def step(self, action):
        return self.env.step(action)
    
    def close(self):
        self.env.close()


class TestHierarchicalRL(unittest.TestCase):
    """Test cases for HierarchicalRL."""
    
    def setUp(self):
        """Set up test environment."""
        self.env = TestCartPoleEnv()
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up after tests."""
        self.env.close()
        shutil.rmtree(self.temp_dir)
    
    def test_init(self):
        """Test initialization of HierarchicalRL."""
        agent = HierarchicalRL(
            env=self.env,
            num_options=2,
            meta_model_type="PPO",
            option_model_type="PPO"
        )
        
        self.assertEqual(agent.num_options, 2)
        self.assertEqual(agent.meta_model_type, "PPO")
        self.assertEqual(agent.option_model_type, "PPO")
        self.assertIsNone(agent.meta_controller)
        self.assertEqual(len(agent.options), 0)
    
    def test_initialize_models(self):
        """Test model initialization."""
        agent = HierarchicalRL(
            env=self.env,
            num_options=2,
            meta_model_type="PPO",
            option_model_type="PPO"
        )
        
        agent.initialize_models()
        
        self.assertIsNotNone(agent.meta_controller)
        self.assertEqual(len(agent.options), 2)
    
    @patch('scripts.advanced_rl.PPO')
    def test_execute_option(self, mock_ppo):
        """Test option execution."""
        # Mock predict method
        mock_instance = MagicMock()
        mock_instance.predict.return_value = (0, None)
        mock_ppo.return_value = mock_instance
        
        agent = HierarchicalRL(
            env=self.env,
            num_options=2,
            meta_model_type="PPO",
            option_model_type="PPO",
            option_duration=5
        )
        
        agent.initialize_models()
        
        # Mock env.step to always return the same values
        self.env.step = MagicMock(return_value=(
            np.zeros(4),  # observation
            1.0,          # reward
            False,        # done
            {}            # info
        ))
        
        next_obs, total_reward, done, steps, info = agent.execute_option(0, np.zeros(4))
        
        # Option should execute for option_duration steps
        self.assertEqual(steps, 5)
        self.assertEqual(total_reward, 5.0)  # 1.0 reward per step
        self.assertFalse(done)
        
        # Test early termination
        self.env.step = MagicMock(return_value=(
            np.zeros(4),  # observation
            1.0,          # reward
            True,         # done
            {}            # info
        ))
        
        next_obs, total_reward, done, steps, info = agent.execute_option(0, np.zeros(4))
        
        # Option should terminate early
        self.assertEqual(steps, 1)
        self.assertEqual(total_reward, 1.0)
        self.assertTrue(done)
    
    def test_save_load(self):
        """Test saving and loading."""
        # Create and initialize agent
        agent = HierarchicalRL(
            env=self.env,
            num_options=2,
            meta_model_type="PPO",
            option_model_type="PPO"
        )
        agent.initialize_models()
        
        # Save agent
        save_path = os.path.join(self.temp_dir, "test_agent")
        agent.save(save_path)
        
        # Check that files were created
        self.assertTrue(os.path.exists(os.path.join(save_path, "meta_controller")))
        self.assertTrue(os.path.exists(os.path.join(save_path, "option_0")))
        self.assertTrue(os.path.exists(os.path.join(save_path, "option_1")))
        self.assertTrue(os.path.exists(os.path.join(save_path, "params.json")))
        self.assertTrue(os.path.exists(os.path.join(save_path, "history.json")))
        
        # Load agent
        loaded_agent = HierarchicalRL.load(save_path, self.env)
        
        # Check that parameters were loaded correctly
        self.assertEqual(loaded_agent.num_options, 2)
        self.assertEqual(loaded_agent.meta_model_type, "PPO")
        self.assertEqual(loaded_agent.option_model_type, "PPO")
        self.assertIsNotNone(loaded_agent.meta_controller)
        self.assertEqual(len(loaded_agent.options), 2)


class TestMultiAgentMarketSystem(unittest.TestCase):
    """Test cases for MultiAgentMarketSystem."""
    
    def setUp(self):
        """Set up test environment."""
        self.env = TestCartPoleEnv()
        
    def tearDown(self):
        """Clean up after tests."""
        self.env.close()
    
    def test_init(self):
        """Test initialization of MultiAgentMarketSystem."""
        system = MultiAgentMarketSystem(
            base_env=self.env,
            num_agents=3,
            agent_types=["PPO", "A2C", "PPO"]
        )
        
        self.assertEqual(system.num_agents, 3)
        self.assertEqual(system.agent_types, ["PPO", "A2C", "PPO"])
        self.assertEqual(len(system.agents), 3)
    
    @patch('scripts.advanced_rl.PPO')
    @patch('scripts.advanced_rl.A2C')
    def test_step(self, mock_a2c, mock_ppo):
        """Test stepping in multi-agent system."""
        # Mock predict methods
        mock_ppo_instance = MagicMock()
        mock_ppo_instance.predict.return_value = (0, None)
        mock_ppo.return_value = mock_ppo_instance
        
        mock_a2c_instance = MagicMock()
        mock_a2c_instance.predict.return_value = (1, None)
        mock_a2c.return_value = mock_a2c_instance
        
        system = MultiAgentMarketSystem(
            base_env=self.env,
            num_agents=2,
            agent_types=["PPO", "A2C"]
        )
        
        # Mock env.step
        self.env.step = MagicMock(return_value=(
            np.zeros(4),  # observation
            [1.0, 2.0],   # rewards for each agent
            False,        # done
            {}            # info
        ))
        
        obs = np.zeros(4)
        next_obs, rewards, done, info = system.step(obs)
        
        # Check that agents made different actions
        self.assertEqual(mock_ppo_instance.predict.call_count, 1)
        self.assertEqual(mock_a2c_instance.predict.call_count, 1)
        
        # Check that the environment step was called
        self.env.step.assert_called_once()
        
        # Check the returned values
        self.assertEqual(len(rewards), 2)
        self.assertTrue(np.all(next_obs == np.zeros(4)))
        self.assertFalse(done)
    
    def test_reset(self):
        """Test resetting the multi-agent system."""
        system = MultiAgentMarketSystem(
            base_env=self.env,
            num_agents=2,
            agent_types=["PPO", "A2C"]
        )
        
        # Mock env.reset
        self.env.reset = MagicMock(return_value=np.zeros(4))
        
        obs = system.reset()
        
        # Check that environment reset was called
        self.env.reset.assert_called_once()
        
        # Check that the episode rewards were reset
        for agent_rewards in system.history['episode_rewards'].values():
            self.assertEqual(len(agent_rewards), 1)
            self.assertEqual(agent_rewards[-1], 0)


class TestImitationLearning(unittest.TestCase):
    """Test cases for ImitationLearning."""
    
    def setUp(self):
        """Set up test environment."""
        self.env = TestCartPoleEnv()
        self.temp_dir = tempfile.mkdtemp()
        
        # Create simple expert data
        self.expert_data = {
            'observations': np.random.rand(100, 4),
            'actions': np.random.randint(0, 2, 100)
        }
        
    def tearDown(self):
        """Clean up after tests."""
        self.env.close()
        shutil.rmtree(self.temp_dir)
    
    def test_init(self):
        """Test initialization of ImitationLearning."""
        model = ImitationLearning(
            env=self.env,
            expert_data=self.expert_data,
            method='behavioral_cloning'
        )
        
        self.assertEqual(model.method, 'behavioral_cloning')
        self.assertEqual(model.model_type, 'mlp')
        self.assertEqual(model.expert_data, self.expert_data)
        self.assertIsNone(model.policy_network)
    
    @patch('scripts.advanced_rl.tf.keras.Sequential')
    def test_create_policy_network(self, mock_sequential):
        """Test creation of policy network."""
        mock_model = MagicMock()
        mock_sequential.return_value = mock_model
        
        model = ImitationLearning(
            env=self.env,
            expert_data=self.expert_data,
            method='behavioral_cloning'
        )
        
        network = model._create_policy_network()
        
        # Check that Sequential was called
        mock_sequential.assert_called_once()
        
        # Check that the model was compiled
        mock_model.compile.assert_called_once()
    
    @patch('scripts.advanced_rl.ImitationLearning._create_policy_network')
    @patch('scripts.advanced_rl.ImitationLearning._preprocess_expert_data')
    def test_train_behavioral_cloning(self, mock_preprocess, mock_create_network):
        """Test training with behavioral cloning."""
        # Mock preprocessing
        mock_preprocess.return_value = (np.random.rand(100, 4), np.random.randint(0, 2, 100))
        
        # Mock network
        mock_model = MagicMock()
        mock_history = MagicMock()
        mock_history.history = {
            'loss': [0.5, 0.4, 0.3],
            'val_accuracy': [0.6, 0.7, 0.8]
        }
        mock_model.fit.return_value = mock_history
        mock_create_network.return_value = mock_model
        
        model = ImitationLearning(
            env=self.env,
            expert_data=self.expert_data,
            method='behavioral_cloning'
        )
        
        history = model.train_behavioral_cloning(epochs=3)
        
        # Check that the network was created
        mock_create_network.assert_called_once()
        
        # Check that fit was called
        mock_model.fit.assert_called_once()
        
        # Check that history was updated
        self.assertEqual(model.history['loss'], [0.5, 0.4, 0.3])
        self.assertEqual(model.history['validation_accuracy'], [0.6, 0.7, 0.8])
    
    def test_save_load(self):
        """Test saving and loading ImitationLearning model."""
        # Create a model with a simple network
        model = ImitationLearning(
            env=self.env,
            expert_data=self.expert_data,
            method='behavioral_cloning'
        )
        
        # Create a simple Keras model
        policy_network = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(4,)),
            tf.keras.layers.Dense(10, activation='relu'),
            tf.keras.layers.Dense(2, activation='softmax')
        ])
        policy_network.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        model.policy_network = policy_network
        
        # Update history
        model.history['loss'] = [0.5, 0.4, 0.3]
        model.history['validation_accuracy'] = [0.6, 0.7, 0.8]
        
        # Save model
        save_path = os.path.join(self.temp_dir, "test_model")
        model.save(save_path)
        
        # Check that files were created
        self.assertTrue(os.path.exists(os.path.join(save_path, "policy_network")))
        self.assertTrue(os.path.exists(os.path.join(save_path, "history.json")))
        self.assertTrue(os.path.exists(os.path.join(save_path, "params.json")))
        
        # Load model
        loaded_model = ImitationLearning.load(save_path, self.env)
        
        # Check that parameters were loaded correctly
        self.assertEqual(loaded_model.method, 'behavioral_cloning')
        self.assertEqual(loaded_model.model_type, 'mlp')
        self.assertIsNotNone(loaded_model.policy_network)
        self.assertEqual(loaded_model.history['loss'], [0.5, 0.4, 0.3])
        self.assertEqual(loaded_model.history['validation_accuracy'], [0.6, 0.7, 0.8])


if __name__ == "__main__":
    unittest.main() 
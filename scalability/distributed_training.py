import os
import time
import uuid
import logging
import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import torch.multiprocessing as mp
from stable_baselines3 import PPO, A2C, SAC
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.sac import SACConfig
from ray.rllib.algorithms.a2c import A2CConfig
from ray.tune.integration.mlflow import MLflowLoggerCallback
from typing import Dict, List, Optional, Any, Union, Callable, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DistributedTraining:
    """
    Distributed training implementation for reinforcement learning algorithms.
    Supports multiple backends: Ray, PyTorch DDP, and TensorFlow Distribution Strategy.
    """
    
    def __init__(
        self,
        backend: str = "ray",
        num_workers: int = 4,
        num_gpus: int = 0,
        log_dir: str = "logs/distributed",
        model_dir: str = "models/distributed",
        use_mlflow: bool = False,
        mlflow_tracking_uri: Optional[str] = None,
        mlflow_experiment_name: str = "distributed_rl"
    ):
        """
        Initialize distributed training.
        
        Args:
            backend: The distributed backend to use ('ray', 'pytorch', 'tensorflow')
            num_workers: Number of worker processes or actors
            num_gpus: Number of GPUs to use
            log_dir: Directory for logs
            model_dir: Directory for saved models
            use_mlflow: Whether to use MLflow for tracking
            mlflow_tracking_uri: MLflow tracking URI
            mlflow_experiment_name: MLflow experiment name
        """
        self.backend = backend
        self.num_workers = num_workers
        self.num_gpus = num_gpus
        self.log_dir = log_dir
        self.model_dir = model_dir
        self.use_mlflow = use_mlflow
        self.mlflow_tracking_uri = mlflow_tracking_uri
        self.mlflow_experiment_name = mlflow_experiment_name
        
        # Create directories if they don't exist
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)
        
        # Initialize distributed backend
        self._init_backend()
        
    def _init_backend(self):
        """Initialize the selected distributed backend."""
        if self.backend == "ray":
            # Initialize Ray
            if not ray.is_initialized():
                ray.init(num_cpus=self.num_workers, num_gpus=self.num_gpus)
            logger.info(f"Ray initialized with {self.num_workers} CPUs and {self.num_gpus} GPUs")
            
            # Initialize MLflow if requested
            if self.use_mlflow:
                try:
                    import mlflow
                    if self.mlflow_tracking_uri:
                        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
                    mlflow.set_experiment(self.mlflow_experiment_name)
                    logger.info(f"MLflow initialized with experiment: {self.mlflow_experiment_name}")
                except ImportError:
                    logger.warning("MLflow not installed. MLflow tracking disabled.")
                    self.use_mlflow = False
                    
        elif self.backend == "pytorch":
            # Will initialize in train() method using multiprocessing
            logger.info(f"PyTorch DDP backend selected with {self.num_workers} workers")
            
        elif self.backend == "tensorflow":
            # Set up TensorFlow distribution strategy
            if self.num_gpus > 0:
                self.strategy = tf.distribute.MirroredStrategy()
            else:
                self.strategy = tf.distribute.MultiWorkerMirroredStrategy()
                
            logger.info(f"TensorFlow distribution strategy initialized: {self.strategy.__class__.__name__}")
            
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")
    
    def train_ray(
        self,
        env_name: str,
        algorithm: str = "PPO",
        config: Optional[Dict[str, Any]] = None,
        total_timesteps: int = 1000000,
        tune_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Train using Ray RLlib.
        
        Args:
            env_name: Gym environment name
            algorithm: RL algorithm to use ('PPO', 'SAC', 'A2C')
            config: Algorithm configuration
            total_timesteps: Total timesteps to train for
            tune_config: Ray Tune configuration for hyperparameter tuning
            
        Returns:
            Dict: Training results
        """
        # Set up default configuration
        if config is None:
            config = {}
            
        # Create base algorithm config
        if algorithm == "PPO":
            algo_config = PPOConfig()
        elif algorithm == "SAC":
            algo_config = SACConfig()
        elif algorithm == "A2C":
            algo_config = A2CConfig()
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
            
        # Update with user configuration
        algo_config = algo_config.environment(env_name)
        algo_config = algo_config.resources(num_gpus=self.num_gpus / self.num_workers)
        algo_config = algo_config.rollouts(num_rollout_workers=self.num_workers - 1)
        algo_config = algo_config.framework("torch")  # Could make this configurable
        
        # Apply custom configurations
        for key, value in config.items():
            algo_config = algo_config.training(**{key: value})
            
        # Setup training
        if tune_config is not None:
            # Use Ray Tune for hyperparameter tuning
            tune_callbacks = []
            if self.use_mlflow:
                tune_callbacks.append(MLflowLoggerCallback(
                    tracking_uri=self.mlflow_tracking_uri,
                    experiment_name=self.mlflow_experiment_name,
                    save_artifact=True
                ))
                
            # Set up parameter search space
            config_dict = algo_config.to_dict()
            for param, space in tune_config.items():
                config_dict[param] = space
                
            # Run hyperparameter tuning
            analysis = tune.run(
                algorithm,
                config=config_dict,
                stop={"training_iteration": total_timesteps},
                num_samples=tune_config.get("num_samples", 10),
                local_dir=self.log_dir,
                callbacks=tune_callbacks
            )
            
            # Return best trial results
            best_trial = analysis.get_best_trial("episode_reward_mean", "max")
            best_config = best_trial.config
            logger.info(f"Best hyperparameters found: {best_config}")
            
            # Train final model with best config
            algo = algo_config.resources(num_gpus=self.num_gpus).build()
            for key, value in best_config.items():
                if key in algo.config:
                    algo.config[key] = value
                    
            algo.train(total_timesteps // algo.config.train_batch_size)
            checkpoint_path = algo.save(self.model_dir)
            
            return {
                "best_config": best_config,
                "best_reward": best_trial.last_result["episode_reward_mean"],
                "checkpoint_path": checkpoint_path
            }
        else:
            # Direct training without hyperparameter tuning
            algo = algo_config.build()
            
            # Train for specified timesteps
            iteration_count = total_timesteps // algo.config.train_batch_size
            results = None
            
            for i in range(iteration_count):
                results = algo.train()
                
                if i % 10 == 0:
                    logger.info(f"Iteration {i}/{iteration_count}: Mean reward = {results['episode_reward_mean']:.2f}")
                    
                    # Save checkpoint periodically
                    if i % 50 == 0:
                        checkpoint_path = algo.save(os.path.join(self.model_dir, f"checkpoint_{i}"))
                        logger.info(f"Saved checkpoint to {checkpoint_path}")
            
            # Save final model
            checkpoint_path = algo.save(self.model_dir)
            logger.info(f"Training completed. Final checkpoint saved to {checkpoint_path}")
            
            # Return training results
            return {
                "mean_reward": results["episode_reward_mean"],
                "checkpoint_path": checkpoint_path,
                "final_results": results
            }
    
    def _setup_pytorch_model(self, env_name, algorithm, config):
        """Set up a PyTorch model for distributed training."""
        # Create a dummy environment to get observation and action spaces
        import gym
        env = gym.make(env_name)
        
        # Create appropriate model based on algorithm
        if algorithm == "PPO":
            model = PPO("MlpPolicy", env, **config)
        elif algorithm == "SAC":
            model = SAC("MlpPolicy", env, **config)
        elif algorithm == "A2C":
            model = A2C("MlpPolicy", env, **config)
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
            
        env.close()
        return model
    
    def _pytorch_worker(
        self,
        rank: int,
        world_size: int,
        env_name: str,
        algorithm: str,
        config: Dict[str, Any],
        total_timesteps: int,
        model_dir: str,
        log_dir: str
    ):
        """Worker function for PyTorch distributed training."""
        # Set up process group
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        dist.init_process_group("gloo", rank=rank, world_size=world_size)
        
        # Set device
        if torch.cuda.is_available() and self.num_gpus > 0:
            device = torch.device(f"cuda:{rank % self.num_gpus}")
        else:
            device = torch.device("cpu")
            
        torch.cuda.set_device(device)
        
        # Create model
        model = self._setup_pytorch_model(env_name, algorithm, config)
        
        # Extract the policy network for DDP
        policy_net = model.policy.to(device)
        ddp_policy = DistributedDataParallel(policy_net, device_ids=[device])
        
        # Replace the original policy network with DDP version
        model.policy = ddp_policy
        
        # Adjust learning rate based on number of processes
        if hasattr(model, 'learning_rate'):
            model.learning_rate = model.learning_rate * world_size
            
        # Train the model
        model.learn(total_timesteps=total_timesteps // world_size)
        
        # Save model on rank 0
        if rank == 0:
            save_path = os.path.join(model_dir, f"{algorithm}_{int(time.time())}")
            model.save(save_path)
            logger.info(f"Model saved to {save_path}")
            
        # Clean up
        dist.destroy_process_group()
    
    def train_pytorch(
        self,
        env_name: str,
        algorithm: str = "PPO",
        config: Optional[Dict[str, Any]] = None,
        total_timesteps: int = 1000000
    ) -> Dict[str, Any]:
        """
        Train using PyTorch DDP.
        
        Args:
            env_name: Gym environment name
            algorithm: RL algorithm to use ('PPO', 'SAC', 'A2C')
            config: Algorithm configuration
            total_timesteps: Total timesteps to train for
            
        Returns:
            Dict: Training results
        """
        if config is None:
            config = {}
            
        # Add current time to model directory to avoid conflicts
        timestamp = int(time.time())
        model_dir = os.path.join(self.model_dir, f"{timestamp}")
        os.makedirs(model_dir, exist_ok=True)
        
        # Launch distributed training
        mp.spawn(
            self._pytorch_worker,
            args=(
                self.num_workers,
                env_name,
                algorithm,
                config,
                total_timesteps,
                model_dir,
                self.log_dir
            ),
            nprocs=self.num_workers,
            join=True
        )
        
        # Return info about training
        return {
            "model_dir": model_dir,
            "algorithm": algorithm,
            "env_name": env_name,
            "num_workers": self.num_workers,
            "total_timesteps": total_timesteps
        }
    
    def train_tensorflow(
        self,
        env_name: str,
        algorithm: str = "PPO",
        config: Optional[Dict[str, Any]] = None,
        total_timesteps: int = 1000000
    ) -> Dict[str, Any]:
        """
        Train using TensorFlow distribution strategy.
        
        Args:
            env_name: Gym environment name
            algorithm: RL algorithm to use ('PPO', 'SAC', 'A2C')
            config: Algorithm configuration
            total_timesteps: Total timesteps to train for
            
        Returns:
            Dict: Training results
        """
        # This is a simplified implementation as TF distributed RL
        # would require more complex adaptation of the algorithms
        with self.strategy.scope():
            # Create model
            import gym
            env = gym.make(env_name)
            
            # TensorFlow implementation would go here
            # For now, we'll log a message and return
            logger.warning("TensorFlow distributed training not fully implemented.")
            
            env.close()
            
        return {
            "status": "not_fully_implemented",
            "backend": "tensorflow",
            "model_dir": self.model_dir
        }
    
    def train(
        self,
        env_name: str,
        algorithm: str = "PPO",
        config: Optional[Dict[str, Any]] = None,
        total_timesteps: int = 1000000,
        tune_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Train a reinforcement learning model using distributed computing.
        
        Args:
            env_name: Gym environment name
            algorithm: RL algorithm to use ('PPO', 'SAC', 'A2C')
            config: Algorithm configuration
            total_timesteps: Total timesteps to train for
            tune_config: Configuration for hyperparameter tuning
            
        Returns:
            Dict: Training results
        """
        logger.info(f"Starting distributed training with {self.backend} backend")
        logger.info(f"Environment: {env_name}, Algorithm: {algorithm}")
        
        # Use appropriate backend
        if self.backend == "ray":
            return self.train_ray(env_name, algorithm, config, total_timesteps, tune_config)
        elif self.backend == "pytorch":
            return self.train_pytorch(env_name, algorithm, config, total_timesteps)
        elif self.backend == "tensorflow":
            return self.train_tensorflow(env_name, algorithm, config, total_timesteps)
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")
    
    def shutdown(self):
        """Shut down the distributed backend."""
        if self.backend == "ray" and ray.is_initialized():
            ray.shutdown()
            logger.info("Ray shutdown complete")
            
    def __del__(self):
        """Clean up resources on deletion."""
        try:
            self.shutdown()
        except:
            pass


@ray.remote
class DistributedReplayBuffer:
    """
    Distributed replay buffer implementation using Ray.
    Allows multiple workers to share experience for off-policy algorithms.
    """
    
    def __init__(self, capacity: int = 1000000, alpha: float = 0.6, beta: float = 0.4):
        """
        Initialize distributed replay buffer.
        
        Args:
            capacity: Maximum capacity of the buffer
            alpha: Priority exponent (0 = uniform sampling)
            beta: Importance sampling exponent
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.buffer = []
        self.priorities = []
        self.position = 0
        self.size = 0
        
    def add(self, experience: Tuple):
        """
        Add an experience to the buffer.
        
        Args:
            experience: Experience tuple (state, action, reward, next_state, done)
        """
        max_priority = max(self.priorities) if self.priorities else 1.0
        
        if self.size < self.capacity:
            self.buffer.append(experience)
            self.priorities.append(max_priority)
            self.size += 1
        else:
            self.buffer[self.position] = experience
            self.priorities[self.position] = max_priority
            
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size: int) -> Tuple[List, List[int], np.ndarray]:
        """
        Sample a batch of experiences from the buffer with prioritization.
        
        Args:
            batch_size: Number of experiences to sample
            
        Returns:
            Tuple: (sampled experiences, indices, importance sampling weights)
        """
        if self.size < batch_size:
            batch_size = self.size
            
        # Priority sampling
        priorities = np.array(self.priorities)
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        
        # Sample indices
        indices = np.random.choice(self.size, batch_size, replace=False, p=probabilities)
        
        # Get experiences
        experiences = [self.buffer[idx] for idx in indices]
        
        # Calculate importance sampling weights
        weights = (self.size * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()  # Normalize
        
        return experiences, indices, weights
        
    def update_priorities(self, indices: List[int], priorities: List[float]):
        """
        Update priorities for sampled experiences.
        
        Args:
            indices: Indices of experiences
            priorities: New priorities
        """
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
            
    def get_size(self) -> int:
        """
        Get current size of the buffer.
        
        Returns:
            int: Current size
        """
        return self.size
    
    def reset(self):
        """Reset the buffer."""
        self.buffer = []
        self.priorities = []
        self.position = 0
        self.size = 0
            

def create_distributed_training(
    backend: str = "ray",
    num_workers: int = 4,
    num_gpus: int = 0,
    use_mlflow: bool = False
) -> DistributedTraining:
    """
    Factory function to create distributed training instance.
    
    Args:
        backend: The distributed backend to use ('ray', 'pytorch', 'tensorflow')
        num_workers: Number of worker processes
        num_gpus: Number of GPUs to use
        use_mlflow: Whether to use MLflow for tracking
        
    Returns:
        DistributedTraining: Instance of distributed training
    """
    return DistributedTraining(
        backend=backend,
        num_workers=num_workers,
        num_gpus=num_gpus,
        use_mlflow=use_mlflow
    ) 
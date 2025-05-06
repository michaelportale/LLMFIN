import numpy as np
import tensorflow as tf

# Just use the stub-based approach which works better with IDEs
from scripts.tensorflow_keras_stubs import Callback, K

import matplotlib.pyplot as plt
import os
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau as TorchReduceLROnPlateau
import pandas as pd
from stable_baselines3.common.callbacks import BaseCallback as SB3Callback
import json
import time

class CyclicLearningRateScheduler(Callback):
    """
    Cyclic learning rate scheduler for Keras models.
    Implements the cyclical learning rate policy described in the paper
    "Cyclical Learning Rates for Training Neural Networks" by Leslie N. Smith.
    """
    
    def __init__(self, base_lr=1e-4, max_lr=1e-2, step_size=2000, mode='triangular', gamma=1.0, scale_fn=None):
        """
        Initialize the cyclic learning rate scheduler.
        
        Args:
            base_lr (float): Lower bound of learning rate
            max_lr (float): Upper bound of learning rate
            step_size (int): Size of half a cycle (in batches)
            mode (str): One of {triangular, triangular2, exp_range}
            gamma (float): Constant for exp_range mode
            scale_fn (function): Custom scaling function
        """
        super().__init__()
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma
        
        # Scale function for different modes
        if scale_fn is None:
            if mode == 'triangular':
                self.scale_fn = lambda x: 1.0
                self.scale_mode = 'cycle'
            elif mode == 'triangular2':
                self.scale_fn = lambda x: 1.0 / (2.0 ** (x - 1))
                self.scale_mode = 'cycle'
            elif mode == 'exp_range':
                self.scale_fn = lambda x: gamma ** x
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = 'iterations'
            
        self.clr_iterations = 0
        self.history = {}
        
    def _compute_lr(self):
        """
        Compute the learning rate based on current iteration.
        
        Returns:
            float: Current learning rate
        """
        cycle = np.floor(1 + self.clr_iterations / (2 * self.step_size))
        x = np.abs(self.clr_iterations / self.step_size - 2 * cycle + 1)
        
        if self.scale_mode == 'cycle':
            scale_factor = self.scale_fn(cycle)
        else:
            scale_factor = self.scale_fn(self.clr_iterations)
            
        lr = self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x)) * scale_factor
        
        return lr
    
    def on_train_begin(self, logs=None):
        """
        Initialize on training start.
        """
        logs = logs or {}
        
        if self.clr_iterations == 0:
            K.set_value(self.model.optimizer.lr, self.base_lr)
        else:
            K.set_value(self.model.optimizer.lr, self._compute_lr())
    
    def on_batch_end(self, batch, logs=None):
        """
        Update learning rate after each batch.
        """
        logs = logs or {}
        
        self.clr_iterations += 1
        lr = self._compute_lr()
        K.set_value(self.model.optimizer.lr, lr)
        
        # Record learning rate in history
        self.history.setdefault('lr', []).append(lr)
        self.history.setdefault('iterations', []).append(self.clr_iterations)
        
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)
    
    def plot_lr(self, save_path=None):
        """
        Plot the learning rate schedule.
        
        Args:
            save_path (str): Path to save the plot
        """
        plt.figure(figsize=(10, 6))
        plt.plot(self.history['iterations'], self.history['lr'])
        plt.xlabel('Iterations')
        plt.ylabel('Learning Rate')
        plt.title('Cyclic Learning Rate Schedule')
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    
    def save_history(self, path):
        """
        Save learning rate history.
        
        Args:
            path (str): Path to save the history
        """
        with open(path, 'w') as f:
            json.dump(self.history, f)


class AdaptiveLearningRateScheduler(Callback):
    """
    Adaptive learning rate scheduler that adjusts the learning rate based on
    validation performance.
    """
    
    def __init__(self, monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, 
                 cooldown=0, min_delta=1e-4, verbose=1, mode='min'):
        """
        Initialize the adaptive learning rate scheduler.
        
        Args:
            monitor (str): Quantity to monitor
            factor (float): Factor by which to reduce learning rate
            patience (int): Number of epochs with no improvement after which LR is reduced
            min_lr (float): Lower bound on the learning rate
            cooldown (int): Number of epochs to wait before resuming normal operation
            min_delta (float): Threshold for measuring improvement
            verbose (int): Verbosity mode
            mode (str): One of {min, max} - whether to monitor improvement by minimizing or maximizing
        """
        super().__init__()
        self.monitor = monitor
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.cooldown = cooldown
        self.min_delta = min_delta
        self.verbose = verbose
        self.mode = mode
        
        self.best = np.inf if mode == 'min' else -np.inf
        self.cooldown_counter = 0
        self.wait = 0
        self.history = {}
        
    def on_train_begin(self, logs=None):
        """
        Initialize on training start.
        """
        logs = logs or {}
        self.history = {}
        
    def on_epoch_end(self, epoch, logs=None):
        """
        Check for improvement and update learning rate.
        """
        logs = logs or {}
        current_lr = float(K.get_value(self.model.optimizer.lr))
        
        # Record learning rate in history
        self.history.setdefault('lr', []).append(current_lr)
        self.history.setdefault('epoch', []).append(epoch)
        
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)
        
        # Check if monitor metric is available
        if self.monitor not in logs:
            if self.verbose > 0:
                print(f"Warning: {self.monitor} not in logs")
            return
        
        current = logs.get(self.monitor)
        
        # Determine if improvement
        if self.mode == 'min':
            is_improvement = current < (self.best - self.min_delta)
        else:
            is_improvement = current > (self.best + self.min_delta)
        
        # Update best metric
        if is_improvement:
            self.best = current
            self.wait = 0
        else:
            if self.cooldown_counter > 0:
                self.cooldown_counter -= 1
                self.wait = 0
            else:
                self.wait += 1
                
        # Reduce learning rate if patience is exceeded
        if self.wait >= self.patience:
            if current_lr > self.min_lr:
                new_lr = current_lr * self.factor
                new_lr = max(new_lr, self.min_lr)
                K.set_value(self.model.optimizer.lr, new_lr)
                
                if self.verbose > 0:
                    print(f"\nEpoch {epoch}: Reducing learning rate to {new_lr:.6f}")
                    
                self.wait = 0
                self.cooldown_counter = self.cooldown
                
    def plot_lr(self, save_path=None):
        """
        Plot the learning rate schedule.
        
        Args:
            save_path (str): Path to save the plot
        """
        plt.figure(figsize=(10, 6))
        plt.plot(self.history['epoch'], self.history['lr'])
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Adaptive Learning Rate Schedule')
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()


class CosineAnnealingScheduler(Callback):
    """
    Cosine annealing learning rate scheduler with optional warm restarts.
    """
    
    def __init__(self, initial_lr=1e-3, min_lr=1e-6, cycles=1, cycle_length=10, warmup_epochs=0):
        """
        Initialize the cosine annealing scheduler.
        
        Args:
            initial_lr (float): Initial learning rate
            min_lr (float): Minimum learning rate
            cycles (int): Number of cycles
            cycle_length (int): Length of each cycle in epochs
            warmup_epochs (int): Number of warmup epochs with linear learning rate increase
        """
        super().__init__()
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.cycles = cycles
        self.cycle_length = cycle_length
        self.warmup_epochs = warmup_epochs
        self.history = {}
    
    def _compute_lr(self, epoch):
        """
        Compute the learning rate for a given epoch.
        
        Args:
            epoch (int): Current epoch
            
        Returns:
            float: Learning rate
        """
        if epoch < self.warmup_epochs:
            # Linear warmup
            return self.min_lr + (self.initial_lr - self.min_lr) * (epoch / self.warmup_epochs)
        
        # Adjust epoch to account for warmup
        adjusted_epoch = epoch - self.warmup_epochs
        
        # Determine which cycle we're in
        if self.cycles == 1:
            # Simple cosine annealing without restarts
            progress = adjusted_epoch / self.cycle_length
            progress = min(1.0, progress)
        else:
            # Cosine annealing with warm restarts
            cycle_progress = adjusted_epoch % self.cycle_length
            progress = cycle_progress / self.cycle_length
        
        # Cosine function from 0 to pi gives a nice curve from 1 to 0
        cosine_value = 0.5 * (1 + np.cos(np.pi * progress))
        
        # Scale between min_lr and initial_lr
        return self.min_lr + (self.initial_lr - self.min_lr) * cosine_value
    
    def on_epoch_begin(self, epoch, logs=None):
        """
        Update learning rate at beginning of each epoch.
        """
        lr = self._compute_lr(epoch)
        K.set_value(self.model.optimizer.lr, lr)
        
        # Record learning rate in history
        self.history.setdefault('lr', []).append(lr)
        self.history.setdefault('epoch', []).append(epoch)
    
    def plot_schedule(self, epochs=None, save_path=None):
        """
        Plot the complete learning rate schedule.
        
        Args:
            epochs (int): Number of epochs to plot
            save_path (str): Path to save the plot
        """
        if epochs is None:
            epochs = self.warmup_epochs + self.cycles * self.cycle_length
            
        lrs = [self._compute_lr(epoch) for epoch in range(epochs)]
        
        plt.figure(figsize=(10, 6))
        plt.plot(range(epochs), lrs)
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Cosine Annealing Learning Rate Schedule')
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()


class AdaptiveLearningRateCallbackRL(SB3Callback):
    """
    Adaptive learning rate callback for Stable Baselines3 models.
    Adjusts the learning rate based on episode rewards.
    """
    
    def __init__(self, monitor='reward', factor=0.5, patience=3, min_lr=1e-6, verbose=1):
        """
        Initialize the adaptive learning rate callback.
        
        Args:
            monitor (str): What to monitor ('reward' or 'loss')
            factor (float): Factor by which to reduce learning rate
            patience (int): Number of evaluations with no improvement
            min_lr (float): Lower bound on the learning rate
            verbose (int): Verbosity mode
        """
        super().__init__()
        self.monitor = monitor
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.verbose = verbose
        
        self.best_mean_reward = -np.inf
        self.wait = 0
        self.original_lr = None
        self.current_lr = None
        self.history = {'lr': [], 'step': [], 'reward': []}
    
    def _init_callback(self) -> None:
        """
        Initialize callback with the RL model.
        """
        # Get original learning rate
        self.original_lr = self.model.learning_rate
        self.current_lr = self.original_lr
    
    def _on_step(self) -> bool:
        """
        Method called at each step of training.
        
        Returns:
            bool: Whether training should continue
        """
        # Only adjust LR periodically (every N episodes)
        if self.num_timesteps % self.model.n_steps == 0:
            # Track learning rate
            self.history['lr'].append(self.current_lr)
            self.history['step'].append(self.num_timesteps)
            
            # Get mean episode reward
            mean_reward = np.mean([ep_info["r"] for ep_info in self.model.ep_info_buffer])
            self.history['reward'].append(mean_reward)
            
            # Check if reward has improved
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                self.wait = 0
            else:
                self.wait += 1
                
                # Reduce learning rate if patience is exceeded
                if self.wait >= self.patience:
                    if self.current_lr > self.min_lr:
                        self.current_lr *= self.factor
                        self.current_lr = max(self.current_lr, self.min_lr)
                        
                        # Update model's learning rate
                        self.model.learning_rate = self.current_lr
                        
                        if self.verbose > 0:
                            print(f"\nStep {self.num_timesteps}: Reducing learning rate to {self.current_lr:.8f}")
                            
                        self.wait = 0
                        
        return True
    
    def plot_lr_history(self, save_path=None):
        """
        Plot the learning rate history.
        
        Args:
            save_path (str): Path to save the plot
        """
        plt.figure(figsize=(10, 6))
        
        # Plot learning rate
        plt.subplot(2, 1, 1)
        plt.plot(self.history['step'], self.history['lr'])
        plt.xlabel('Steps')
        plt.ylabel('Learning Rate')
        plt.title('Adaptive Learning Rate History')
        plt.grid(True)
        
        # Plot rewards
        plt.subplot(2, 1, 2)
        plt.plot(self.history['step'], self.history['reward'])
        plt.xlabel('Steps')
        plt.ylabel('Mean Reward')
        plt.title('Reward History')
        plt.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    
    def save_history(self, path):
        """
        Save learning rate history.
        
        Args:
            path (str): Path to save the history
        """
        df = pd.DataFrame(self.history)
        df.to_csv(path, index=False)


class AdaptiveWarmupLinearLRScheduler:
    """
    Custom learning rate scheduler with adaptive warmup and linear decay.
    Automatically adjusts warmup length based on gradient statistics.
    """
    
    def __init__(self, optimizer, initial_lr=1e-3, min_lr=1e-6, warmup_steps=500, 
                 decay_steps=10000, auto_warmup=True, grad_threshold=0.01):
        """
        Initialize the adaptive warmup linear LR scheduler.
        
        Args:
            optimizer: PyTorch optimizer
            initial_lr (float): Initial learning rate after warmup
            min_lr (float): Minimum learning rate
            warmup_steps (int): Number of warmup steps (if auto_warmup is False)
            decay_steps (int): Number of decay steps after warmup
            auto_warmup (bool): Whether to automatically adjust warmup length
            grad_threshold (float): Gradient variance threshold for ending warmup
        """
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps
        self.auto_warmup = auto_warmup
        self.grad_threshold = grad_threshold
        
        self.step_count = 0
        self.warmup_completed = False
        self.gradient_history = []
        self.lr_history = []
        
    def step(self, gradient_stats=None):
        """
        Update learning rate based on step count and gradient statistics.
        
        Args:
            gradient_stats (dict): Dictionary with gradient statistics (variance)
        """
        self.step_count += 1
        
        # Store gradient statistics if provided
        if gradient_stats is not None and 'variance' in gradient_stats:
            self.gradient_history.append(gradient_stats['variance'])
        
        # Determine current phase (warmup or decay)
        if not self.warmup_completed:
            if self.auto_warmup and len(self.gradient_history) >= 50:
                # Check if gradient variance has stabilized
                recent_variance = np.mean(self.gradient_history[-20:])
                if recent_variance < self.grad_threshold:
                    self.warmup_completed = True
                    # Set actual warmup steps to current step count
                    self.warmup_steps = self.step_count
                    print(f"Auto warmup completed after {self.warmup_steps} steps")
            elif self.step_count >= self.warmup_steps:
                self.warmup_completed = True
        
        # Calculate learning rate based on current phase
        if not self.warmup_completed:
            # Linear warmup
            lr = self.min_lr + (self.initial_lr - self.min_lr) * (self.step_count / self.warmup_steps)
        else:
            # Linear decay
            steps_since_warmup = self.step_count - self.warmup_steps
            decay_fraction = max(0, 1 - steps_since_warmup / self.decay_steps)
            lr = self.min_lr + (self.initial_lr - self.min_lr) * decay_fraction
        
        # Update learning rate for all parameter groups
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        # Store learning rate history
        self.lr_history.append(lr)
        
        return lr
    
    def get_last_lr(self):
        """
        Return the last computed learning rate.
        
        Returns:
            float: Last learning rate
        """
        return self.lr_history[-1] if self.lr_history else self.min_lr
    
    def plot_lr_history(self, save_path=None):
        """
        Plot the learning rate history.
        
        Args:
            save_path (str): Path to save the plot
        """
        plt.figure(figsize=(10, 6))
        
        plt.plot(range(len(self.lr_history)), self.lr_history)
        plt.axvline(x=self.warmup_steps, color='r', linestyle='--', 
                   label=f'Warmup End ({self.warmup_steps} steps)')
        
        plt.xlabel('Steps')
        plt.ylabel('Learning Rate')
        plt.title('Adaptive Warmup Linear Decay Learning Rate Schedule')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    
    def plot_gradient_history(self, save_path=None):
        """
        Plot the gradient variance history.
        
        Args:
            save_path (str): Path to save the plot
        """
        if not self.gradient_history:
            print("No gradient history to plot")
            return
            
        plt.figure(figsize=(10, 6))
        
        plt.plot(range(len(self.gradient_history)), self.gradient_history)
        plt.axvline(x=self.warmup_steps, color='r', linestyle='--', 
                   label=f'Warmup End ({self.warmup_steps} steps)')
        plt.axhline(y=self.grad_threshold, color='g', linestyle='--',
                   label=f'Gradient Threshold ({self.grad_threshold})')
        
        plt.xlabel('Steps')
        plt.ylabel('Gradient Variance')
        plt.title('Gradient Variance History')
        plt.legend()
        plt.grid(True)
        plt.yscale('log')
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()


def create_lr_scheduler(model_type, scheduler_type, **kwargs):
    """
    Factory function to create learning rate schedulers based on model type.
    
    Args:
        model_type (str): Type of model ('keras', 'torch', 'stable_baselines3')
        scheduler_type (str): Type of scheduler
        **kwargs: Additional arguments for the scheduler
        
    Returns:
        object: Learning rate scheduler
    """
    if model_type == 'keras':
        if scheduler_type == 'cyclic':
            return CyclicLearningRateScheduler(**kwargs)
        elif scheduler_type == 'adaptive':
            return AdaptiveLearningRateScheduler(**kwargs)
        elif scheduler_type == 'cosine':
            return CosineAnnealingScheduler(**kwargs)
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")
            
    elif model_type == 'torch':
        if scheduler_type == 'adaptive':
            # Get optimizer
            optimizer = kwargs.pop('optimizer', None)
            if optimizer is None:
                raise ValueError("Optimizer is required for PyTorch schedulers")
            
            return TorchReduceLROnPlateau(optimizer, **kwargs)
        elif scheduler_type == 'adaptive_warmup':
            # Get optimizer
            optimizer = kwargs.pop('optimizer', None)
            if optimizer is None:
                raise ValueError("Optimizer is required for PyTorch schedulers")
                
            return AdaptiveWarmupLinearLRScheduler(optimizer, **kwargs)
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")
            
    elif model_type == 'stable_baselines3':
        if scheduler_type == 'adaptive':
            return AdaptiveLearningRateCallbackRL(**kwargs)
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")
            
    else:
        raise ValueError(f"Unknown model type: {model_type}")


class LearningRateMonitor:
    """
    Monitor and log learning rates during training.
    """
    
    def __init__(self, log_dir=None):
        """
        Initialize the learning rate monitor.
        
        Args:
            log_dir (str): Directory to save logs
        """
        self.log_dir = log_dir
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            
        self.history = {
            'timestamp': [],
            'model': [],
            'lr': [],
            'step': [],
            'metric': []
        }
    
    def log(self, model_name, lr, step=None, metric=None):
        """
        Log a learning rate value.
        
        Args:
            model_name (str): Name of the model
            lr (float): Learning rate value
            step (int): Training step or epoch
            metric (float): Performance metric value
        """
        self.history['timestamp'].append(time.time())
        self.history['model'].append(model_name)
        self.history['lr'].append(lr)
        self.history['step'].append(step)
        self.history['metric'].append(metric)
        
        # Save to log file if directory is provided
        if self.log_dir:
            log_file = os.path.join(self.log_dir, f"{model_name}_lr_log.csv")
            pd.DataFrame(self.history).to_csv(log_file, index=False)
    
    def plot(self, models=None, save_path=None):
        """
        Plot learning rate history for specified models.
        
        Args:
            models (list): List of model names to plot
            save_path (str): Path to save the plot
        """
        df = pd.DataFrame(self.history)
        
        if models:
            df = df[df['model'].isin(models)]
            
        # Group by model
        models = df['model'].unique()
        
        plt.figure(figsize=(12, 8))
        
        for model in models:
            model_data = df[df['model'] == model]
            plt.plot(model_data['step'], model_data['lr'], label=model)
            
        plt.xlabel('Step/Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate History')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    
    def correlate_with_metric(self, model, save_path=None):
        """
        Plot learning rate correlation with performance metric.
        
        Args:
            model (str): Model name
            save_path (str): Path to save the plot
        """
        df = pd.DataFrame(self.history)
        model_data = df[df['model'] == model]
        
        # Check if metric data is available
        if model_data['metric'].isna().all():
            print(f"No metric data available for model {model}")
            return
            
        plt.figure(figsize=(12, 8))
        
        # Plot learning rate vs metric as scatter plot
        plt.scatter(model_data['lr'], model_data['metric'], alpha=0.7)
        
        plt.xlabel('Learning Rate')
        plt.ylabel('Performance Metric')
        plt.title(f'Learning Rate vs Performance for {model}')
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show() 
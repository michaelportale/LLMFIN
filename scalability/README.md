# Scalability Module

This module provides scalability features for machine learning and reinforcement learning applications, enabling them to handle larger workloads, efficiently utilize computational resources, and operate in production environments.

## Components

### 1. Distributed Training (`distributed_training.py`)

The distributed training component enables training of machine learning models across multiple machines or processors, significantly reducing training time for large models.

Key features:
- Support for multiple backends (Ray, PyTorch DDP, TensorFlow)
- Configurable worker management
- Hyperparameter tuning integration
- Model checkpointing and saving
- Performance monitoring

Usage example:
```python
from scalability.distributed_training import create_distributed_training

# Create distributed training instance
trainer = create_distributed_training(
    backend="ray",
    num_workers=4,
    num_gpus=1
)

# Train model
results = trainer.train(
    env_name="CartPole-v1",
    algorithm="PPO",
    total_timesteps=1000000
)

# Shutdown
trainer.shutdown()
```

### 2. Model Serving (`model_serving.py`)

The model serving component provides infrastructure for deploying trained models in production environments, handling inference requests efficiently.

Key features:
- Multiple serving backends (Flask, Ray Serve, TensorFlow Serving, MLflow)
- Support for different model types (stable-baselines3, RLlib, custom)
- Batch inference for higher throughput
- GPU acceleration
- Health monitoring endpoints

Usage example:
```python
from scalability.model_serving import create_model_server

# Create and start model server
server = create_model_server(
    model_path="models/ppo_cartpole.zip",
    backend="flask",
    model_type="stable_baselines",
    port=8000
)

server.start()

# Server will run in background
# To stop:
# server.stop()
```

### 3. Data Caching (`data_caching.py`)

The data caching component speeds up applications by storing frequently accessed data in memory or other fast storage, reducing computation and I/O overhead.

Key features:
- Multiple cache types (LRU, Redis, File-based)
- Time-based expiration
- Support for large datasets with memory mapping
- Thread-safe operations
- Specialized dataset caching with preprocessing

Usage example:
```python
from scalability.data_caching import create_cache, cache_decorator

# Create a cache
cache = create_cache(cache_type="lru", capacity=100, ttl=3600)

# Use cache decorator
@cache_decorator(cache=cache)
def expensive_computation(x, y):
    # ... complex calculation
    return result

# Result will be cached for future calls with same arguments
result = expensive_computation(10, 20)
```

### 4. Background Job Processing (`background_jobs.py`)

The background job processing component enables long-running tasks to execute asynchronously without blocking the main application.

Key features:
- Priority-based job queue
- Worker pool with configurable size
- Job status tracking and persistence
- Timeout handling
- Callback support for completed/failed jobs

Usage example:
```python
from scalability.background_jobs import create_job_manager

# Create and start job manager
job_manager = create_job_manager(
    num_workers=4,
    persistence_enabled=True,
    storage_dir="job_data"
)

job_manager.start()

# Enqueue a long-running job
def training_task(model_type, dataset, epochs):
    # ... long-running machine learning training
    return trained_model

job_id = job_manager.enqueue_job(
    func=training_task,
    args=("neural_network", "large_dataset.csv", 100),
    timeout=3600,  # 1 hour timeout
    priority=10,
    description="Training neural network model"
)

# Check job status later
job = job_manager.get_job(job_id)
print(f"Job status: {job.status}, Progress: {job.progress}%")

# Shutdown job manager when done
job_manager.stop()
```

## Integration

These components can be used independently or together to create a complete scalable machine learning system:

1. Use distributed training to train models faster
2. Cache datasets to speed up training and evaluation
3. Run resource-intensive tasks as background jobs
4. Deploy trained models with the model serving infrastructure

The modular design allows for flexible adoption based on specific application needs. 
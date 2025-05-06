"""
Scalability module for enhancing ML and RL systems with distributed capabilities.

This module provides components for distributed training, model serving,
data caching, and background job processing to help ML applications scale.
"""

# Version info
__version__ = '0.1.0'

# Import factory functions to make them available at the module level
from scalability.distributed_training import create_distributed_training
from scalability.model_serving import create_model_server, BatchInferenceProcessor
from scalability.data_caching import create_cache, cache_decorator, DatasetCache
from scalability.background_jobs import create_job_manager, JobStatus

# Define what's available when using import *
__all__ = [
    'create_distributed_training',
    'create_model_server',
    'BatchInferenceProcessor',
    'create_cache',
    'cache_decorator',
    'DatasetCache',
    'create_job_manager',
    'JobStatus',
] 
import os
import time
import pickle
import json
import hashlib
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Union, Optional, Callable, Tuple
import logging
import threading
import redis
from functools import lru_cache, wraps
from collections import OrderedDict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LRUCache:
    """
    Simple LRU (Least Recently Used) cache implementation with time-based expiration.
    """
    
    def __init__(self, capacity: int = 100, ttl: int = 3600):
        """
        Initialize LRU cache.
        
        Args:
            capacity: Maximum number of items to store
            ttl: Time to live in seconds
        """
        self.capacity = capacity
        self.ttl = ttl
        self.cache = OrderedDict()
        self.timestamps = {}
        self.lock = threading.RLock()
        
    def get(self, key):
        """
        Get item from cache.
        
        Args:
            key: Cache key
            
        Returns:
            value: Cached value or None if not found or expired
        """
        with self.lock:
            if key not in self.cache:
                return None
                
            # Check expiration
            current_time = time.time()
            if current_time - self.timestamps[key] > self.ttl:
                # Expired
                self.cache.pop(key)
                self.timestamps.pop(key)
                return None
                
            # Update position (most recently used)
            value = self.cache.pop(key)
            self.cache[key] = value
            self.timestamps[key] = current_time
            
            return value
    
    def put(self, key, value):
        """
        Put item in cache.
        
        Args:
            key: Cache key
            value: Value to cache
        """
        with self.lock:
            if key in self.cache:
                # Update existing key
                self.cache.pop(key)
            elif len(self.cache) >= self.capacity:
                # Remove least recently used item
                self.cache.popitem(last=False)
                
            # Add new item
            self.cache[key] = value
            self.timestamps[key] = time.time()
    
    def remove(self, key):
        """
        Remove item from cache.
        
        Args:
            key: Cache key
        """
        with self.lock:
            if key in self.cache:
                self.cache.pop(key)
                self.timestamps.pop(key)
    
    def clear(self):
        """Clear the cache."""
        with self.lock:
            self.cache.clear()
            self.timestamps.clear()
    
    def get_stats(self):
        """
        Get cache statistics.
        
        Returns:
            Dict: Cache statistics
        """
        with self.lock:
            return {
                "size": len(self.cache),
                "capacity": self.capacity,
                "ttl": self.ttl,
                "keys": list(self.cache.keys())
            }


class RedisCache:
    """
    Redis-based distributed cache for shared data across processes/machines.
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        ttl: int = 3600,
        prefix: str = "rl_cache:"
    ):
        """
        Initialize Redis cache.
        
        Args:
            host: Redis host
            port: Redis port
            db: Redis database
            password: Redis password
            ttl: Default time to live in seconds
            prefix: Key prefix for namespace isolation
        """
        self.ttl = ttl
        self.prefix = prefix
        
        # Connect to Redis
        self.redis = redis.Redis(
            host=host,
            port=port,
            db=db,
            password=password,
            decode_responses=False  # Keep binary data for pickle
        )
        
        try:
            # Test connection
            self.redis.ping()
            logger.info(f"Connected to Redis at {host}:{port}")
        except redis.ConnectionError as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise
    
    def _make_key(self, key):
        """
        Create prefixed key.
        
        Args:
            key: Original key
            
        Returns:
            str: Prefixed key
        """
        if isinstance(key, str):
            return f"{self.prefix}{key}"
        else:
            # For non-string keys, use a hash
            hash_obj = hashlib.md5(str(key).encode())
            return f"{self.prefix}{hash_obj.hexdigest()}"
    
    def get(self, key):
        """
        Get item from cache.
        
        Args:
            key: Cache key
            
        Returns:
            value: Cached value or None if not found
        """
        prefixed_key = self._make_key(key)
        
        try:
            data = self.redis.get(prefixed_key)
            if data is None:
                return None
                
            # Deserialize
            return pickle.loads(data)
        except Exception as e:
            logger.error(f"Error getting key {key} from Redis: {e}")
            return None
    
    def put(self, key, value, ttl: Optional[int] = None):
        """
        Put item in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (or default if None)
        """
        prefixed_key = self._make_key(key)
        ttl = ttl if ttl is not None else self.ttl
        
        try:
            # Serialize
            data = pickle.dumps(value)
            
            # Store with expiration
            self.redis.setex(prefixed_key, ttl, data)
        except Exception as e:
            logger.error(f"Error setting key {key} in Redis: {e}")
    
    def remove(self, key):
        """
        Remove item from cache.
        
        Args:
            key: Cache key
        """
        prefixed_key = self._make_key(key)
        
        try:
            self.redis.delete(prefixed_key)
        except Exception as e:
            logger.error(f"Error removing key {key} from Redis: {e}")
    
    def clear(self, pattern: str = "*"):
        """
        Clear keys matching pattern.
        
        Args:
            pattern: Pattern to match (default: all keys in namespace)
        """
        try:
            # Get all keys matching pattern
            keys = self.redis.keys(f"{self.prefix}{pattern}")
            
            if keys:
                # Delete all matching keys
                self.redis.delete(*keys)
        except Exception as e:
            logger.error(f"Error clearing Redis cache: {e}")
    
    def get_stats(self):
        """
        Get cache statistics.
        
        Returns:
            Dict: Cache statistics
        """
        try:
            # Get info from Redis
            info = self.redis.info()
            
            # Count keys in namespace
            keys = self.redis.keys(f"{self.prefix}*")
            
            return {
                "size": len(keys),
                "memory_used": info.get("used_memory_human", "N/A"),
                "redis_version": info.get("redis_version", "N/A"),
                "ttl": self.ttl,
                "prefix": self.prefix
            }
        except Exception as e:
            logger.error(f"Error getting Redis stats: {e}")
            return {
                "error": str(e)
            }


class FileCache:
    """
    File-based cache for larger datasets with memory mapping capabilities.
    """
    
    def __init__(
        self,
        cache_dir: str = "cache",
        ttl: int = 86400,  # 24 hours
        use_mmap: bool = True
    ):
        """
        Initialize file cache.
        
        Args:
            cache_dir: Directory for cache files
            ttl: Time to live in seconds
            use_mmap: Whether to use memory mapping for large files
        """
        self.cache_dir = cache_dir
        self.ttl = ttl
        self.use_mmap = use_mmap
        self.index_file = os.path.join(cache_dir, "index.json")
        self.index = {}
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
        
        # Load index if it exists
        self._load_index()
        
        # Start cleanup thread
        self._start_cleanup_thread()
    
    def _load_index(self):
        """Load index from file."""
        if os.path.exists(self.index_file):
            try:
                with open(self.index_file, 'r') as f:
                    self.index = json.load(f)
            except Exception as e:
                logger.error(f"Error loading cache index: {e}")
                self.index = {}
    
    def _save_index(self):
        """Save index to file."""
        try:
            with open(self.index_file, 'w') as f:
                json.dump(self.index, f)
        except Exception as e:
            logger.error(f"Error saving cache index: {e}")
    
    def _hash_key(self, key):
        """
        Create hash of key for filename.
        
        Args:
            key: Cache key
            
        Returns:
            str: Hashed key
        """
        hash_obj = hashlib.md5(str(key).encode())
        return hash_obj.hexdigest()
    
    def _start_cleanup_thread(self):
        """Start background thread for cleanup."""
        def cleanup_loop():
            while True:
                try:
                    self._cleanup_expired()
                except Exception as e:
                    logger.error(f"Error during cache cleanup: {e}")
                    
                # Sleep for an hour
                time.sleep(3600)
                
        thread = threading.Thread(target=cleanup_loop, daemon=True)
        thread.start()
    
    def _cleanup_expired(self):
        """Remove expired cache entries."""
        current_time = time.time()
        expired_keys = []
        
        # Find expired entries
        for key, entry in self.index.items():
            if current_time - entry["timestamp"] > entry["ttl"]:
                expired_keys.append(key)
                
        # Remove expired entries
        for key in expired_keys:
            self.remove(key)
            
        if expired_keys:
            logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")
    
    def get(self, key):
        """
        Get item from cache.
        
        Args:
            key: Cache key
            
        Returns:
            value: Cached value or None if not found or expired
        """
        hashed_key = self._hash_key(key)
        
        # Check if key exists in index
        if hashed_key not in self.index:
            return None
            
        entry = self.index[hashed_key]
        
        # Check expiration
        if time.time() - entry["timestamp"] > entry["ttl"]:
            # Expired
            self.remove(hashed_key)
            return None
            
        # Get file path
        file_path = os.path.join(self.cache_dir, entry["filename"])
        
        if not os.path.exists(file_path):
            # File missing
            self.index.pop(hashed_key)
            self._save_index()
            return None
            
        try:
            # Load based on data type
            data_type = entry["type"]
            
            if data_type == "pickle":
                with open(file_path, "rb") as f:
                    return pickle.load(f)
                    
            elif data_type == "numpy":
                if self.use_mmap:
                    return np.load(file_path, mmap_mode="r")
                else:
                    return np.load(file_path)
                    
            elif data_type == "pandas":
                if file_path.endswith(".csv"):
                    return pd.read_csv(file_path)
                elif file_path.endswith(".parquet"):
                    return pd.read_parquet(file_path)
                else:
                    return pd.read_pickle(file_path)
                    
            else:
                # Unknown type
                logger.warning(f"Unknown data type: {data_type}")
                return None
                
        except Exception as e:
            logger.error(f"Error loading cached data: {e}")
            return None
            
        finally:
            # Update timestamp (touch)
            entry["timestamp"] = time.time()
            self.index[hashed_key] = entry
            self._save_index()
    
    def put(self, key, value, ttl: Optional[int] = None):
        """
        Put item in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (or default if None)
        """
        hashed_key = self._hash_key(key)
        ttl = ttl if ttl is not None else self.ttl
        
        # Determine data type and file extension
        if isinstance(value, np.ndarray):
            data_type = "numpy"
            file_ext = ".npy"
        elif isinstance(value, pd.DataFrame):
            data_type = "pandas"
            file_ext = ".parquet"
        else:
            data_type = "pickle"
            file_ext = ".pkl"
            
        # Create filename
        filename = f"{hashed_key}{file_ext}"
        file_path = os.path.join(self.cache_dir, filename)
        
        try:
            # Save based on data type
            if data_type == "numpy":
                np.save(file_path, value)
                
            elif data_type == "pandas":
                if file_ext == ".parquet":
                    value.to_parquet(file_path)
                else:
                    value.to_pickle(file_path)
                    
            else:
                with open(file_path, "wb") as f:
                    pickle.dump(value, f)
                    
            # Update index
            self.index[hashed_key] = {
                "filename": filename,
                "type": data_type,
                "timestamp": time.time(),
                "ttl": ttl
            }
            
            self._save_index()
            
        except Exception as e:
            logger.error(f"Error caching data: {e}")
    
    def remove(self, key):
        """
        Remove item from cache.
        
        Args:
            key: Cache key
        """
        hashed_key = self._hash_key(key)
        
        if hashed_key in self.index:
            try:
                # Get file path
                filename = self.index[hashed_key]["filename"]
                file_path = os.path.join(self.cache_dir, filename)
                
                # Remove file if it exists
                if os.path.exists(file_path):
                    os.remove(file_path)
                    
                # Remove from index
                self.index.pop(hashed_key)
                self._save_index()
                
            except Exception as e:
                logger.error(f"Error removing cached data: {e}")
    
    def clear(self):
        """Clear the cache."""
        try:
            # Remove all cache files
            for entry in self.index.values():
                file_path = os.path.join(self.cache_dir, entry["filename"])
                if os.path.exists(file_path):
                    os.remove(file_path)
                    
            # Clear index
            self.index = {}
            self._save_index()
            
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
    
    def get_stats(self):
        """
        Get cache statistics.
        
        Returns:
            Dict: Cache statistics
        """
        current_time = time.time()
        total_size = 0
        active_entries = 0
        
        for hashed_key, entry in self.index.items():
            file_path = os.path.join(self.cache_dir, entry["filename"])
            
            if os.path.exists(file_path):
                total_size += os.path.getsize(file_path)
                
                # Check if active (not expired)
                if current_time - entry["timestamp"] <= entry["ttl"]:
                    active_entries += 1
        
        return {
            "total_entries": len(self.index),
            "active_entries": active_entries,
            "total_size_bytes": total_size,
            "total_size_mb": total_size / (1024 * 1024),
            "cache_dir": self.cache_dir
        }


def cache_decorator(cache=None, ttl=None, key_fn=None):
    """
    Decorator for caching function results.
    
    Args:
        cache: Cache object (must support get/put methods)
        ttl: Time to live for cached results
        key_fn: Function to generate cache key (default: use args and kwargs)
        
    Returns:
        Function decorator
    """
    # Use in-memory LRU cache if no cache provided
    if cache is None:
        cache = LRUCache()
        
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            if key_fn is not None:
                cache_key = key_fn(*args, **kwargs)
            else:
                # Default: use function name, args, and sorted kwargs
                key_parts = [func.__name__]
                key_parts.extend([str(arg) for arg in args])
                key_parts.extend([f"{k}={v}" for k, v in sorted(kwargs.items())])
                cache_key = ":".join(key_parts)
            
            # Try to get from cache
            cached_result = cache.get(cache_key)
            if cached_result is not None:
                return cached_result
                
            # Call function
            result = func(*args, **kwargs)
            
            # Cache result
            cache.put(cache_key, result, ttl)
            
            return result
            
        return wrapper
        
    return decorator


def create_cache(cache_type="lru", **kwargs):
    """
    Factory function to create a cache.
    
    Args:
        cache_type: Type of cache ('lru', 'redis', 'file')
        **kwargs: Additional arguments for the specific cache
        
    Returns:
        Cache object
    """
    if cache_type == "lru":
        return LRUCache(**kwargs)
    elif cache_type == "redis":
        return RedisCache(**kwargs)
    elif cache_type == "file":
        return FileCache(**kwargs)
    else:
        raise ValueError(f"Unsupported cache type: {cache_type}")


class DatasetCache:
    """
    Specialized cache for ML datasets with preprocessing capabilities.
    """
    
    def __init__(
        self,
        base_cache=None,
        preprocess_fn: Optional[Callable] = None,
        cache_dir: str = "dataset_cache"
    ):
        """
        Initialize dataset cache.
        
        Args:
            base_cache: Underlying cache implementation
            preprocess_fn: Function for preprocessing datasets
            cache_dir: Directory for cached datasets
        """
        self.preprocess_fn = preprocess_fn
        
        # Create file cache if no base cache provided
        if base_cache is None:
            self.cache = FileCache(cache_dir=cache_dir)
        else:
            self.cache = base_cache
    
    def get_dataset(self, dataset_id, preprocess=True, **kwargs):
        """
        Get dataset, either from cache or by loading and preprocessing.
        
        Args:
            dataset_id: Dataset identifier
            preprocess: Whether to preprocess the dataset
            **kwargs: Additional arguments for dataset loading
            
        Returns:
            dataset: The requested dataset
        """
        # Try to get from cache
        cache_key = f"dataset:{dataset_id}:{preprocess}"
        cached_dataset = self.cache.get(cache_key)
        
        if cached_dataset is not None:
            logger.info(f"Dataset {dataset_id} loaded from cache")
            return cached_dataset
            
        # Need to load and possibly preprocess
        try:
            # Load dataset (implementation depends on dataset type)
            dataset = self._load_dataset(dataset_id, **kwargs)
            
            # Preprocess if requested and function available
            if preprocess and self.preprocess_fn is not None:
                dataset = self.preprocess_fn(dataset)
                
            # Cache for future use
            self.cache.put(cache_key, dataset)
            
            return dataset
            
        except Exception as e:
            logger.error(f"Error loading dataset {dataset_id}: {e}")
            raise
    
    def _load_dataset(self, dataset_id, **kwargs):
        """
        Load dataset implementation.
        Override this in subclasses for specific dataset types.
        
        Args:
            dataset_id: Dataset identifier
            **kwargs: Additional arguments for dataset loading
            
        Returns:
            dataset: The loaded dataset
        """
        # Basic implementation - override in subclasses
        if dataset_id.endswith('.csv'):
            return pd.read_csv(dataset_id, **kwargs)
        elif dataset_id.endswith('.parquet'):
            return pd.read_parquet(dataset_id, **kwargs)
        elif dataset_id.endswith('.npy'):
            return np.load(dataset_id, **kwargs)
        else:
            raise ValueError(f"Unsupported dataset format: {dataset_id}")
    
    def invalidate(self, dataset_id=None):
        """
        Invalidate cache for specific dataset or all datasets.
        
        Args:
            dataset_id: Dataset identifier or None for all
        """
        if dataset_id is None:
            # Clear all dataset cache
            self.cache.clear()
        else:
            # Clear specific dataset
            for preprocess in [True, False]:
                cache_key = f"dataset:{dataset_id}:{preprocess}"
                self.cache.remove(cache_key)
                
    def get_stats(self):
        """
        Get cache statistics.
        
        Returns:
            Dict: Cache statistics
        """
        return self.cache.get_stats() 
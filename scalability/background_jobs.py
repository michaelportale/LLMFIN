import os
import time
import uuid
import pickle
import json
import threading
import queue
import logging
import datetime
from typing import Dict, List, Any, Union, Optional, Callable, Tuple
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class JobStatus(Enum):
    """Job status enum."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"

class Job:
    """
    Represents a background job with execution tracking.
    """
    
    def __init__(
        self,
        job_id: str = None,
        func: Callable = None,
        args: Tuple = None,
        kwargs: Dict = None,
        timeout: int = None,
        priority: int = 0,
        description: str = ""
    ):
        """
        Initialize job.
        
        Args:
            job_id: Unique job identifier
            func: Function to execute
            args: Positional arguments for function
            kwargs: Keyword arguments for function
            timeout: Maximum execution time in seconds
            priority: Job priority (higher values = higher priority)
            description: Human-readable job description
        """
        self.job_id = job_id or str(uuid.uuid4())
        self.func = func
        self.args = args or ()
        self.kwargs = kwargs or {}
        self.timeout = timeout
        self.priority = priority
        self.description = description
        
        # Job tracking
        self.status = JobStatus.PENDING
        self.created_at = datetime.datetime.now()
        self.started_at = None
        self.completed_at = None
        self.result = None
        self.error = None
        self.progress = 0
    
    def to_dict(self):
        """
        Convert job to dictionary for serialization.
        
        Returns:
            Dict: Job as dictionary
        """
        return {
            "job_id": self.job_id,
            "description": self.description,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "progress": self.progress,
            "error": str(self.error) if self.error else None,
            "priority": self.priority,
            "timeout": self.timeout
        }
    
    @classmethod
    def from_dict(cls, data):
        """
        Create job from dictionary.
        
        Args:
            data: Dictionary representation of job
            
        Returns:
            Job: Created job
        """
        job = cls(
            job_id=data["job_id"],
            priority=data.get("priority", 0),
            timeout=data.get("timeout"),
            description=data.get("description", "")
        )
        
        # Set status
        job.status = JobStatus(data["status"])
        
        # Set timestamps
        job.created_at = datetime.datetime.fromisoformat(data["created_at"])
        if data.get("started_at"):
            job.started_at = datetime.datetime.fromisoformat(data["started_at"])
        if data.get("completed_at"):
            job.completed_at = datetime.datetime.fromisoformat(data["completed_at"])
            
        # Set progress and error
        job.progress = data.get("progress", 0)
        job.error = data.get("error")
        
        return job

class JobQueue:
    """
    Job queue with priority and persistence capabilities.
    """
    
    def __init__(
        self,
        max_size: int = 1000,
        storage_dir: Optional[str] = None,
        persistence_enabled: bool = False
    ):
        """
        Initialize job queue.
        
        Args:
            max_size: Maximum queue size
            storage_dir: Directory for job storage
            persistence_enabled: Whether to persist jobs to disk
        """
        self.max_size = max_size
        self.storage_dir = storage_dir
        self.persistence_enabled = persistence_enabled
        
        # Priority queue for jobs
        self.queue = queue.PriorityQueue(maxsize=max_size)
        
        # Job tracking
        self.jobs = {}
        self.lock = threading.RLock()
        
        # Initialize storage
        if persistence_enabled and storage_dir:
            os.makedirs(storage_dir, exist_ok=True)
            
            # Load pending jobs from storage
            self._load_pending_jobs()
    
    def _save_job(self, job):
        """
        Save job to storage.
        
        Args:
            job: Job to save
        """
        if not self.persistence_enabled or not self.storage_dir:
            return
            
        try:
            job_path = os.path.join(self.storage_dir, f"{job.job_id}.json")
            with open(job_path, 'w') as f:
                json.dump(job.to_dict(), f)
        except Exception as e:
            logger.error(f"Error saving job {job.job_id}: {e}")
    
    def _load_pending_jobs(self):
        """Load pending jobs from storage."""
        if not self.persistence_enabled or not self.storage_dir:
            return
            
        try:
            # Find all job files
            job_files = [f for f in os.listdir(self.storage_dir) if f.endswith('.json')]
            
            for job_file in job_files:
                try:
                    job_path = os.path.join(self.storage_dir, job_file)
                    with open(job_path, 'r') as f:
                        job_data = json.load(f)
                        
                    # Only load pending jobs
                    if job_data["status"] == JobStatus.PENDING.value:
                        job = Job.from_dict(job_data)
                        
                        # Need to load function from somewhere else
                        # For now, just log that job was found
                        logger.info(f"Found pending job {job.job_id} in storage")
                        
                        # Note: We can't actually enqueue these jobs without the functions
                        # This would need to be handled by the specific implementation
                        
                except Exception as e:
                    logger.error(f"Error loading job from {job_file}: {e}")
        
        except Exception as e:
            logger.error(f"Error loading pending jobs: {e}")
    
    def enqueue(self, job):
        """
        Add job to queue.
        
        Args:
            job: Job to enqueue
            
        Returns:
            str: Job ID
        """
        with self.lock:
            # Check if queue is full
            if self.queue.full():
                raise ValueError("Job queue is full")
                
            # Store job for tracking
            self.jobs[job.job_id] = job
            
            # Add to priority queue (negated priority for highest-first)
            self.queue.put((-job.priority, job.job_id))
            
            # Save job to storage
            self._save_job(job)
            
            logger.info(f"Enqueued job {job.job_id} with priority {job.priority}")
            
            return job.job_id
    
    def dequeue(self):
        """
        Get next job from queue.
        
        Returns:
            Job: Next job or None if queue is empty
        """
        try:
            # Get job ID from priority queue
            _, job_id = self.queue.get(block=False)
            
            with self.lock:
                # Get job from tracking dict
                job = self.jobs.get(job_id)
                
                if job:
                    # Update job status
                    job.status = JobStatus.RUNNING
                    job.started_at = datetime.datetime.now()
                    
                    # Save updated job
                    self._save_job(job)
                    
                return job
                
        except queue.Empty:
            return None
    
    def get_job(self, job_id):
        """
        Get job by ID.
        
        Args:
            job_id: Job ID
            
        Returns:
            Job: Job or None if not found
        """
        with self.lock:
            return self.jobs.get(job_id)
    
    def update_job(self, job_id, status=None, progress=None, result=None, error=None):
        """
        Update job status.
        
        Args:
            job_id: Job ID
            status: New job status
            progress: Job progress (0-100)
            result: Job result
            error: Job error
            
        Returns:
            bool: Whether update was successful
        """
        with self.lock:
            job = self.jobs.get(job_id)
            
            if not job:
                return False
                
            # Update job
            if status is not None:
                job.status = status
                
                # Set completion time if finished
                if status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELED]:
                    job.completed_at = datetime.datetime.now()
                    
            if progress is not None:
                job.progress = min(max(0, progress), 100)
                
            if result is not None:
                job.result = result
                
            if error is not None:
                job.error = error
                
            # Save updated job
            self._save_job(job)
            
            return True
    
    def cancel_job(self, job_id):
        """
        Cancel job.
        
        Args:
            job_id: Job ID
            
        Returns:
            bool: Whether cancellation was successful
        """
        with self.lock:
            job = self.jobs.get(job_id)
            
            if not job or job.status != JobStatus.PENDING:
                return False
                
            # Update job status
            job.status = JobStatus.CANCELED
            job.completed_at = datetime.datetime.now()
            
            # Save updated job
            self._save_job(job)
            
            # Note: We can't remove it from the queue directly
            # It will be discarded when dequeued
            
            return True
    
    def get_queue_stats(self):
        """
        Get queue statistics.
        
        Returns:
            Dict: Queue statistics
        """
        with self.lock:
            # Count jobs by status
            status_counts = {}
            for status in JobStatus:
                status_counts[status.value] = 0
                
            for job in self.jobs.values():
                status_counts[job.status.value] += 1
                
            return {
                "queue_size": self.queue.qsize(),
                "max_size": self.max_size,
                "total_jobs": len(self.jobs),
                "status_counts": status_counts
            }

class WorkerThread(threading.Thread):
    """
    Worker thread for processing jobs.
    """
    
    def __init__(
        self,
        job_queue,
        results_callback=None,
        error_callback=None,
        poll_interval=1,
        name=None
    ):
        """
        Initialize worker thread.
        
        Args:
            job_queue: Job queue
            results_callback: Callback for successful job completion
            error_callback: Callback for job errors
            poll_interval: Queue polling interval in seconds
            name: Thread name
        """
        super().__init__(name=name)
        self.daemon = True
        self.job_queue = job_queue
        self.results_callback = results_callback
        self.error_callback = error_callback
        self.poll_interval = poll_interval
        self.running = True
    
    def run(self):
        """Run worker thread."""
        logger.info(f"Worker thread {self.name} starting")
        
        while self.running:
            # Get next job
            job = self.job_queue.dequeue()
            
            if job is None:
                # No jobs, wait and try again
                time.sleep(self.poll_interval)
                continue
                
            # Skip canceled jobs
            if job.status == JobStatus.CANCELED:
                continue
                
            # Execute job
            try:
                logger.info(f"Executing job {job.job_id}: {job.description}")
                
                # Create timeout context if needed
                if job.timeout:
                    # Using threading timer for timeout
                    timeout_event = threading.Event()
                    timer = threading.Timer(job.timeout, timeout_event.set)
                    timer.daemon = True
                    timer.start()
                    
                    # Check periodically for timeout
                    start_time = time.time()
                    result = None
                    
                    # Start in separate thread to allow monitoring
                    result_container = [None]
                    error_container = [None]
                    
                    def execute_job():
                        try:
                            result_container[0] = job.func(*job.args, **job.kwargs)
                        except Exception as e:
                            error_container[0] = e
                    
                    job_thread = threading.Thread(target=execute_job)
                    job_thread.daemon = True
                    job_thread.start()
                    
                    # Wait for job to complete or timeout
                    while job_thread.is_alive():
                        if timeout_event.is_set():
                            # Timeout occurred
                            error_msg = f"Job timed out after {job.timeout} seconds"
                            self.job_queue.update_job(
                                job.job_id,
                                status=JobStatus.FAILED,
                                error=error_msg
                            )
                            
                            if self.error_callback:
                                self.error_callback(job, error_msg)
                                
                            break
                            
                        job_thread.join(0.1)
                        
                    # Cancel timer if job completed
                    timer.cancel()
                    
                    # Handle result or error
                    if error_container[0] is not None:
                        raise error_container[0]
                        
                    result = result_container[0]
                    
                else:
                    # No timeout, execute directly
                    result = job.func(*job.args, **job.kwargs)
                
                # Update job status and result
                self.job_queue.update_job(
                    job.job_id,
                    status=JobStatus.COMPLETED,
                    progress=100,
                    result=result
                )
                
                # Call results callback if provided
                if self.results_callback:
                    self.results_callback(job, result)
                    
                logger.info(f"Job {job.job_id} completed successfully")
                
            except Exception as e:
                logger.error(f"Error executing job {job.job_id}: {e}")
                
                # Update job status and error
                self.job_queue.update_job(
                    job.job_id,
                    status=JobStatus.FAILED,
                    error=str(e)
                )
                
                # Call error callback if provided
                if self.error_callback:
                    self.error_callback(job, e)
    
    def stop(self):
        """Stop worker thread."""
        self.running = False

class JobManager:
    """
    Manager for background job processing.
    """
    
    def __init__(
        self,
        num_workers: int = 4,
        max_queue_size: int = 1000,
        storage_dir: Optional[str] = None,
        persistence_enabled: bool = False
    ):
        """
        Initialize job manager.
        
        Args:
            num_workers: Number of worker threads
            max_queue_size: Maximum queue size
            storage_dir: Directory for job storage
            persistence_enabled: Whether to persist jobs to disk
        """
        self.num_workers = num_workers
        
        # Create job queue
        self.job_queue = JobQueue(
            max_size=max_queue_size,
            storage_dir=storage_dir,
            persistence_enabled=persistence_enabled
        )
        
        # Create worker threads
        self.workers = []
        
        # Job callbacks
        self.results_callbacks = []
        self.error_callbacks = []
        
        # Running flag
        self.running = False
    
    def add_results_callback(self, callback):
        """
        Add callback for successful job completion.
        
        Args:
            callback: Callback function (job, result) -> None
        """
        self.results_callbacks.append(callback)
    
    def add_error_callback(self, callback):
        """
        Add callback for job errors.
        
        Args:
            callback: Callback function (job, error) -> None
        """
        self.error_callbacks.append(callback)
    
    def _results_callback(self, job, result):
        """Call all results callbacks."""
        for callback in self.results_callbacks:
            try:
                callback(job, result)
            except Exception as e:
                logger.error(f"Error in results callback: {e}")
    
    def _error_callback(self, job, error):
        """Call all error callbacks."""
        for callback in self.error_callbacks:
            try:
                callback(job, error)
            except Exception as e:
                logger.error(f"Error in error callback: {e}")
    
    def start(self):
        """Start job manager."""
        if self.running:
            logger.warning("Job manager is already running")
            return
            
        logger.info(f"Starting job manager with {self.num_workers} workers")
        
        # Create and start worker threads
        for i in range(self.num_workers):
            worker = WorkerThread(
                self.job_queue,
                results_callback=self._results_callback,
                error_callback=self._error_callback,
                name=f"worker-{i}"
            )
            worker.start()
            self.workers.append(worker)
            
        self.running = True
    
    def stop(self):
        """Stop job manager."""
        if not self.running:
            logger.warning("Job manager is not running")
            return
            
        logger.info("Stopping job manager")
        
        # Stop all worker threads
        for worker in self.workers:
            worker.stop()
            
        # Wait for threads to finish
        for worker in self.workers:
            worker.join(timeout=1.0)
            
        self.workers = []
        self.running = False
    
    def enqueue_job(self, func, args=None, kwargs=None, timeout=None, priority=0, job_id=None, description=""):
        """
        Enqueue a job.
        
        Args:
            func: Function to execute
            args: Positional arguments for function
            kwargs: Keyword arguments for function
            timeout: Maximum execution time in seconds
            priority: Job priority (higher values = higher priority)
            job_id: Custom job ID (or auto-generated if None)
            description: Human-readable job description
            
        Returns:
            str: Job ID
        """
        # Create job
        job = Job(
            job_id=job_id,
            func=func,
            args=args,
            kwargs=kwargs,
            timeout=timeout,
            priority=priority,
            description=description
        )
        
        # Enqueue job
        return self.job_queue.enqueue(job)
    
    def get_job(self, job_id):
        """
        Get job by ID.
        
        Args:
            job_id: Job ID
            
        Returns:
            Job: Job or None if not found
        """
        return self.job_queue.get_job(job_id)
    
    def cancel_job(self, job_id):
        """
        Cancel job.
        
        Args:
            job_id: Job ID
            
        Returns:
            bool: Whether cancellation was successful
        """
        return self.job_queue.cancel_job(job_id)
    
    def get_queue_stats(self):
        """
        Get queue statistics.
        
        Returns:
            Dict: Queue statistics
        """
        return self.job_queue.get_queue_stats()

def create_job_manager(
    num_workers: int = 4,
    max_queue_size: int = 1000,
    storage_dir: Optional[str] = None,
    persistence_enabled: bool = False
) -> JobManager:
    """
    Factory function to create a job manager.
    
    Args:
        num_workers: Number of worker threads
        max_queue_size: Maximum queue size
        storage_dir: Directory for job storage
        persistence_enabled: Whether to persist jobs to disk
        
    Returns:
        JobManager: Created job manager
    """
    return JobManager(
        num_workers=num_workers,
        max_queue_size=max_queue_size,
        storage_dir=storage_dir,
        persistence_enabled=persistence_enabled
    ) 
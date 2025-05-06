import os
import time
import numpy as np
import tensorflow as tf
import torch
from typing import Dict, List, Any, Union, Optional, Callable
import logging
import json
import threading
import queue
import flask
from flask import Flask, request, jsonify
from stable_baselines3 import PPO, A2C, SAC
import ray
from ray import serve
import mlflow
import mlflow.pyfunc

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelServing:
    """
    Model serving implementation for deploying trained reinforcement learning models
    in production environments. Supports multiple serving backends including 
    Flask, Ray Serve, and TensorFlow Serving.
    """
    
    def __init__(
        self,
        backend: str = "flask",
        model_path: str = None,
        model_type: str = "stable_baselines",
        framework: str = "pytorch",
        port: int = 8000,
        num_replicas: int = 1,
        batch_size: int = 1,
        use_gpu: bool = False,
        log_dir: str = "logs/serving"
    ):
        """
        Initialize model serving.
        
        Args:
            backend: Serving backend ('flask', 'ray_serve', 'tf_serving', 'mlflow')
            model_path: Path to the saved model
            model_type: Type of model ('stable_baselines', 'rllib', 'custom')
            framework: Framework used ('pytorch', 'tensorflow')
            port: Port for serving
            num_replicas: Number of replicas for distributed serving
            batch_size: Batch size for inference
            use_gpu: Whether to use GPU for inference
            log_dir: Directory for logs
        """
        self.backend = backend
        self.model_path = model_path
        self.model_type = model_type
        self.framework = framework
        self.port = port
        self.num_replicas = num_replicas
        self.batch_size = batch_size
        self.use_gpu = use_gpu
        self.log_dir = log_dir
        
        # Create directories if they don't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # Initialize model
        self.model = None
        if model_path is not None:
            self.load_model()
        
        # Initialize serving components
        self.server = None
        self.is_running = False
        
    def load_model(self):
        """Load the model based on model type and framework."""
        if self.model_path is None:
            raise ValueError("Model path is not provided")
            
        logger.info(f"Loading model from {self.model_path}")
        
        if self.model_type == "stable_baselines":
            # Load Stable Baselines model
            if os.path.isdir(self.model_path):
                model_files = os.listdir(self.model_path)
                zip_files = [f for f in model_files if f.endswith('.zip')]
                if zip_files:
                    self.model_path = os.path.join(self.model_path, zip_files[0])
                    
            if "PPO" in self.model_path:
                self.model = PPO.load(self.model_path)
            elif "A2C" in self.model_path:
                self.model = A2C.load(self.model_path)
            elif "SAC" in self.model_path:
                self.model = SAC.load(self.model_path)
            else:
                # Try to infer model class from file
                try:
                    self.model = PPO.load(self.model_path)
                except:
                    try:
                        self.model = A2C.load(self.model_path)
                    except:
                        try:
                            self.model = SAC.load(self.model_path)
                        except Exception as e:
                            raise ValueError(f"Could not load model: {e}")
                
        elif self.model_type == "rllib":
            # Load RLlib model
            import ray.rllib.algorithms as rllib_algos
            
            if "PPO" in self.model_path:
                self.model = rllib_algos.ppo.PPO.from_checkpoint(self.model_path)
            elif "SAC" in self.model_path:
                self.model = rllib_algos.sac.SAC.from_checkpoint(self.model_path)
            elif "A2C" in self.model_path:
                self.model = rllib_algos.a2c.A2C.from_checkpoint(self.model_path)
            else:
                # Try to infer model class
                try:
                    self.model = rllib_algos.ppo.PPO.from_checkpoint(self.model_path)
                except:
                    try:
                        self.model = rllib_algos.sac.SAC.from_checkpoint(self.model_path)
                    except:
                        try:
                            self.model = rllib_algos.a2c.A2C.from_checkpoint(self.model_path)
                        except Exception as e:
                            raise ValueError(f"Could not load RLlib model: {e}")
                
        elif self.model_type == "mlflow":
            # Load MLflow model
            self.model = mlflow.pyfunc.load_model(self.model_path)
            
        elif self.model_type == "custom":
            # Load custom model - implementation depends on the specific model format
            if self.framework == "pytorch":
                self.model = torch.load(self.model_path)
            elif self.framework == "tensorflow":
                self.model = tf.saved_model.load(self.model_path)
            else:
                raise ValueError(f"Unsupported framework for custom model: {self.framework}")
        
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
            
        logger.info(f"Model loaded successfully: {type(self.model).__name__}")
    
    def predict(self, observation, deterministic=True):
        """
        Make a prediction using the loaded model.
        
        Args:
            observation: Input observation or state
            deterministic: Whether to use deterministic policy
            
        Returns:
            action: Predicted action
        """
        if self.model is None:
            raise ValueError("Model not loaded")
            
        # Convert to numpy array if needed
        if isinstance(observation, list):
            observation = np.array(observation)
            
        # Reshape for batch dimension if needed
        if len(observation.shape) == 1:
            observation = observation.reshape(1, -1)
            
        # Make prediction based on model type
        if self.model_type == "stable_baselines":
            action, _ = self.model.predict(observation, deterministic=deterministic)
            
        elif self.model_type == "rllib":
            if hasattr(self.model, "compute_single_action"):
                action = self.model.compute_single_action(observation[0])
            else:
                action = self.model.compute_actions(observation)[0]
                
        elif self.model_type == "mlflow":
            if hasattr(self.model, "predict"):
                action = self.model.predict(observation)
            else:
                # For MLflow models wrapping stable_baselines
                result = self.model._model_impl({"observations": observation})
                action = result.get("actions", result.get("predictions", None))
                
        elif self.model_type == "custom":
            # Custom model implementation
            if self.framework == "pytorch":
                with torch.no_grad():
                    tensor_input = torch.FloatTensor(observation)
                    if self.use_gpu and torch.cuda.is_available():
                        tensor_input = tensor_input.cuda()
                    action = self.model(tensor_input)
                    if isinstance(action, torch.Tensor):
                        action = action.cpu().numpy()
            
            elif self.framework == "tensorflow":
                action = self.model(tf.convert_to_tensor(observation)).numpy()
                
        return action
    
    def _create_flask_app(self):
        """Create Flask application for serving."""
        app = Flask(__name__)
        
        @app.route('/health', methods=['GET'])
        def health():
            return jsonify({"status": "healthy"})
        
        @app.route('/predict', methods=['POST'])
        def predict_endpoint():
            data = request.json
            
            if not data or 'observation' not in data:
                return jsonify({"error": "Invalid request. 'observation' field is required"}), 400
                
            try:
                observation = data['observation']
                deterministic = data.get('deterministic', True)
                
                # Make prediction
                action = self.predict(observation, deterministic=deterministic)
                
                # Convert to native Python types for JSON serialization
                if isinstance(action, np.ndarray):
                    action = action.tolist()
                    
                return jsonify({"action": action})
                
            except Exception as e:
                logger.error(f"Error during prediction: {e}")
                return jsonify({"error": str(e)}), 500
        
        @app.route('/model_info', methods=['GET'])
        def model_info():
            return jsonify({
                "model_type": self.model_type,
                "framework": self.framework,
                "model_path": self.model_path
            })
            
        return app
    
    def start_flask_server(self):
        """Start Flask server in a separate thread."""
        app = self._create_flask_app()
        
        def run_server():
            app.run(host='0.0.0.0', port=self.port)
            
        self.server_thread = threading.Thread(target=run_server)
        self.server_thread.daemon = True
        self.server_thread.start()
        
        logger.info(f"Flask server started on port {self.port}")
        self.is_running = True
    
    def start_ray_serve(self):
        """Start Ray Serve deployment."""
        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init()
            
        # Initialize Ray Serve if not already initialized
        if not serve.is_initialized():
            serve.start()
            
        # Define deployment
        @serve.deployment(
            name="rl_model_server",
            num_replicas=self.num_replicas,
            route_prefix="/model"
        )
        class RLModelDeployment:
            def __init__(self, model_server):
                self.model_server = model_server
                
            async def __call__(self, request):
                data = await request.json()
                if not data or 'observation' not in data:
                    return {"error": "Invalid request. 'observation' field is required"}
                    
                try:
                    observation = data['observation']
                    deterministic = data.get('deterministic', True)
                    
                    # Make prediction
                    action = self.model_server.predict(observation, deterministic=deterministic)
                    
                    # Convert to native Python types for JSON serialization
                    if isinstance(action, np.ndarray):
                        action = action.tolist()
                        
                    return {"action": action}
                    
                except Exception as e:
                    logger.error(f"Error during prediction: {e}")
                    return {"error": str(e)}
        
        # Create and deploy
        deployment = RLModelDeployment.bind(self)
        serve.run(deployment)
        
        logger.info(f"Ray Serve deployment started")
        self.is_running = True
    
    def start_tf_serving(self):
        """
        Export and start TensorFlow Serving.
        This method requires TensorFlow Serving to be installed separately.
        """
        if self.framework != "tensorflow":
            raise ValueError("TensorFlow Serving requires a TensorFlow model")
            
        # Export model in SavedModel format
        export_path = os.path.join(self.log_dir, "tf_serving_model")
        os.makedirs(export_path, exist_ok=True)
        
        # For models that need special handling
        if self.model_type == "stable_baselines":
            # Extract policy network from stable_baselines
            policy = self.model.policy
            
            # Create a simple serving function
            @tf.function(input_signature=[tf.TensorSpec(shape=[None, None], dtype=tf.float32)])
            def serving_fn(observation):
                # Get action from policy
                action, _, _ = policy.forward(observation, deterministic=True)
                return {"action": action}
                
            # Save the model
            tf.saved_model.save(
                obj={"serving_default": serving_fn},
                export_dir=export_path
            )
        
        elif hasattr(self.model, "save"):
            # For TensorFlow native models
            tf.saved_model.save(self.model, export_path)
            
        # Start TensorFlow Serving (requires tensorflow-serving installed)
        model_name = "rl_model"
        port = self.port
        
        cmd = f"tensorflow_model_server --model_name={model_name} " \
              f"--model_base_path={os.path.abspath(export_path)} " \
              f"--rest_api_port={port} --port={port+1}"
              
        logger.info(f"Starting TensorFlow Serving with command: {cmd}")
        
        # Run in background process
        import subprocess
        self.server_process = subprocess.Popen(
            cmd.split(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Wait for server to start
        time.sleep(5)
        if self.server_process.poll() is not None:
            # Process terminated
            stdout, stderr = self.server_process.communicate()
            logger.error(f"TensorFlow Serving failed to start: {stderr.decode()}")
            raise RuntimeError("TensorFlow Serving failed to start")
            
        logger.info(f"TensorFlow Serving started on port {port}")
        self.is_running = True
    
    def start_mlflow_serving(self):
        """
        Start MLflow model serving.
        Requires the model to be saved in MLflow format.
        """
        if self.model_type != "mlflow":
            # Save current model in MLflow format if it's not already
            if self.model_type == "stable_baselines":
                # Create a wrapper for the stable_baselines model
                class StableBaselinesWrapper(mlflow.pyfunc.PythonModel):
                    def __init__(self, model):
                        self.model = model
                        
                    def predict(self, context, data):
                        observations = data.get("observations", data)
                        deterministic = data.get("deterministic", True)
                        
                        if isinstance(observations, np.ndarray):
                            actions, _ = self.model.predict(observations, deterministic=deterministic)
                        else:
                            # Handle DataFrame or other formats
                            observations = np.array(observations)
                            actions, _ = self.model.predict(observations, deterministic=deterministic)
                            
                        return actions
                
                # Create MLflow model
                wrapped_model = StableBaselinesWrapper(self.model)
                
                # Save to MLflow format
                mlflow_path = os.path.join(self.log_dir, "mlflow_model")
                os.makedirs(mlflow_path, exist_ok=True)
                
                with mlflow.start_run() as run:
                    mlflow.pyfunc.log_model(
                        artifact_path="model",
                        python_model=wrapped_model,
                        code_path=[]  # Add code files if needed
                    )
                    self.model_path = mlflow.get_artifact_uri("model")
                    
                logger.info(f"Model saved in MLflow format at {self.model_path}")
                
                # Update model type
                self.model_type = "mlflow"
                
                # Load the MLflow model
                self.load_model()
            
        # Start MLflow serving
        cmd = f"mlflow models serve -m {self.model_path} -p {self.port} --no-conda"
        if self.num_replicas > 1:
            cmd += f" --workers {self.num_replicas}"
            
        logger.info(f"Starting MLflow serving with command: {cmd}")
        
        # Run in background process
        import subprocess
        self.server_process = subprocess.Popen(
            cmd.split(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Wait for server to start
        time.sleep(5)
        if self.server_process.poll() is not None:
            # Process terminated
            stdout, stderr = self.server_process.communicate()
            logger.error(f"MLflow serving failed to start: {stderr.decode()}")
            raise RuntimeError("MLflow serving failed to start")
            
        logger.info(f"MLflow serving started on port {self.port}")
        self.is_running = True
    
    def start(self):
        """Start the model serving based on the selected backend."""
        if self.is_running:
            logger.warning("Server is already running")
            return
            
        if self.model is None and self.model_path is not None:
            self.load_model()
            
        if self.model is None:
            raise ValueError("Model not loaded")
            
        logger.info(f"Starting model serving with {self.backend} backend")
        
        if self.backend == "flask":
            self.start_flask_server()
        elif self.backend == "ray_serve":
            self.start_ray_serve()
        elif self.backend == "tf_serving":
            self.start_tf_serving()
        elif self.backend == "mlflow":
            self.start_mlflow_serving()
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")
    
    def stop(self):
        """Stop the model serving."""
        if not self.is_running:
            logger.warning("Server is not running")
            return
            
        logger.info(f"Stopping model serving with {self.backend} backend")
        
        if self.backend == "flask":
            # Flask server is running in a daemon thread
            # It will automatically terminate when the program exits
            pass
            
        elif self.backend == "ray_serve":
            # Shutdown Ray Serve
            if serve.is_initialized():
                serve.shutdown()
                
            # Shutdown Ray
            if ray.is_initialized():
                ray.shutdown()
                
        elif self.backend in ["tf_serving", "mlflow"]:
            # Terminate the server process
            if hasattr(self, "server_process"):
                self.server_process.terminate()
                self.server_process.wait()
                
        self.is_running = False
        logger.info("Model serving stopped")
    
    def __del__(self):
        """Clean up resources on deletion."""
        try:
            self.stop()
        except:
            pass


class BatchInferenceProcessor:
    """
    Batch inference processor for efficient processing of large amounts of data.
    Supports parallel processing and batching for throughput optimization.
    """
    
    def __init__(
        self,
        model,
        batch_size: int = 32,
        num_workers: int = 4,
        use_ray: bool = False,
        max_queue_size: int = 1000
    ):
        """
        Initialize batch inference processor.
        
        Args:
            model: Model for inference
            batch_size: Size of batches for processing
            num_workers: Number of parallel workers
            use_ray: Whether to use Ray for parallel processing
            max_queue_size: Maximum size of the queue
        """
        self.model = model
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.use_ray = use_ray
        self.max_queue_size = max_queue_size
        
        # Initialize queue
        self.queue = queue.Queue(maxsize=max_queue_size)
        
        # Initialize workers
        self.workers = []
        self.running = False
        
        # Results storage
        self.results = {}
        self.result_lock = threading.Lock()
        
        # Initialize Ray if needed
        if use_ray and not ray.is_initialized():
            ray.init(num_cpus=num_workers)
    
    def start_workers(self):
        """Start worker threads or Ray actors."""
        if self.running:
            return
            
        self.running = True
        
        if self.use_ray:
            # Define Ray remote function
            @ray.remote
            def process_batch(model, batch_data, batch_ids):
                # Process batch of observations
                results = []
                for data in batch_data:
                    # Get prediction
                    if hasattr(model, "predict"):
                        action, _ = model.predict(data, deterministic=True)
                    else:
                        # Fallback for custom models
                        action = model(data)
                        
                    results.append(action)
                    
                return batch_ids, results
                
            # Store model in Ray's object store
            self.ray_model = ray.put(self.model)
            
        else:
            # Create worker threads
            for _ in range(self.num_workers):
                worker = threading.Thread(target=self._worker_loop)
                worker.daemon = True
                self.workers.append(worker)
                worker.start()
    
    def _worker_loop(self):
        """Worker thread function for processing batches."""
        while self.running:
            try:
                # Get batch from queue with timeout
                batch = self.queue.get(timeout=1.0)
                
                # Process batch
                batch_ids, batch_data = zip(*batch)
                
                # Get predictions
                results = []
                for data in batch_data:
                    # Get prediction
                    if hasattr(self.model, "predict"):
                        action, _ = self.model.predict(data, deterministic=True)
                    else:
                        # Fallback for custom models
                        action = self.model(data)
                        
                    results.append(action)
                
                # Store results
                with self.result_lock:
                    for i, batch_id in enumerate(batch_ids):
                        self.results[batch_id] = results[i]
                
                # Mark task as done
                self.queue.task_done()
                
            except queue.Empty:
                # Queue empty, continue loop
                continue
                
            except Exception as e:
                logger.error(f"Error in worker: {e}")
    
    def process(self, observations: List[np.ndarray]) -> List[np.ndarray]:
        """
        Process a list of observations.
        
        Args:
            observations: List of observations to process
            
        Returns:
            List: Processed results
        """
        if not self.running:
            self.start_workers()
            
        # Generate unique IDs for each observation
        observation_ids = [str(uuid.uuid4()) for _ in range(len(observations))]
        
        # Clear results dictionary
        with self.result_lock:
            self.results = {}
            
        if self.use_ray:
            # Process using Ray
            # Create batches
            batches = []
            current_batch = []
            current_batch_ids = []
            
            for i, (obs_id, obs) in enumerate(zip(observation_ids, observations)):
                current_batch.append(obs)
                current_batch_ids.append(obs_id)
                
                if len(current_batch) >= self.batch_size or i == len(observations) - 1:
                    # Add batch to processing
                    batches.append((current_batch_ids, current_batch))
                    
                    # Reset current batch
                    current_batch = []
                    current_batch_ids = []
            
            # Process batches in parallel
            tasks = [
                process_batch.remote(self.ray_model, batch_data, batch_ids)
                for batch_ids, batch_data in batches
            ]
            
            # Get results
            batch_results = ray.get(tasks)
            
            # Combine results
            result_dict = {}
            for batch_ids, results in batch_results:
                for i, batch_id in enumerate(batch_ids):
                    result_dict[batch_id] = results[i]
                    
            # Order results
            ordered_results = [result_dict[obs_id] for obs_id in observation_ids]
            
            return ordered_results
            
        else:
            # Process using threads
            # Add observations to queue in batches
            batches = []
            current_batch = []
            
            for i, (obs_id, obs) in enumerate(zip(observation_ids, observations)):
                current_batch.append((obs_id, obs))
                
                if len(current_batch) >= self.batch_size or i == len(observations) - 1:
                    # Add batch to queue
                    self.queue.put(current_batch)
                    
                    # Reset current batch
                    current_batch = []
            
            # Wait for queue to be processed
            self.queue.join()
            
            # Get results in order
            ordered_results = []
            with self.result_lock:
                for obs_id in observation_ids:
                    if obs_id in self.results:
                        ordered_results.append(self.results[obs_id])
                    else:
                        # Fallback if result not found
                        logger.warning(f"Result not found for observation {obs_id}")
                        ordered_results.append(None)
                        
            return ordered_results
    
    def stop(self):
        """Stop the batch inference processor."""
        self.running = False
        
        # Wait for threads to complete
        if not self.use_ray:
            for worker in self.workers:
                if worker.is_alive():
                    worker.join(timeout=1.0)
                    
        # Shutdown Ray if needed
        if self.use_ray and ray.is_initialized():
            ray.shutdown()


def create_model_server(
    model_path: str,
    backend: str = "flask",
    model_type: str = "stable_baselines",
    port: int = 8000
) -> ModelServing:
    """
    Factory function to create model serving instance.
    
    Args:
        model_path: Path to the saved model
        backend: Serving backend ('flask', 'ray_serve', 'tf_serving', 'mlflow')
        model_type: Type of model ('stable_baselines', 'rllib', 'custom')
        port: Port for serving
        
    Returns:
        ModelServing: Instance of model serving
    """
    return ModelServing(
        backend=backend,
        model_path=model_path,
        model_type=model_type,
        port=port
    ) 
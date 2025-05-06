#!/usr/bin/env python3
import os
import argparse
import json
import logging
import subprocess
import time
from typing import Dict, Any, List, Optional

logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GCPDeployer:
    """Utility for deploying the RL system on Google Cloud Platform."""
    
    def __init__(self, 
               project_id: str,
               region: str = 'us-central1',
               zone: str = 'us-central1-a',
               cluster_name: str = 'rl-cluster'):
        """Initialize GCP deployer.
        
        Args:
            project_id: GCP project ID
            region: GCP region
            zone: GCP zone
            cluster_name: GKE cluster name
        """
        self.project_id = project_id
        self.region = region
        self.zone = zone
        self.cluster_name = cluster_name
        
        # Check if gcloud is installed
        try:
            subprocess.run(['gcloud', '--version'], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.error("gcloud CLI not found. Please install Google Cloud SDK.")
            raise RuntimeError("gcloud CLI not found")
            
        # Check if kubectl is installed
        try:
            subprocess.run(['kubectl', 'version', '--client'], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.error("kubectl not found. Please install kubectl.")
            raise RuntimeError("kubectl not found")
            
        # Set GCP project
        subprocess.run(['gcloud', 'config', 'set', 'project', project_id], check=True)
        
    def create_artifact_registry(self, 
                               repository_name: str = 'rl-repo',
                               repository_format: str = 'docker') -> Dict[str, Any]:
        """Create Artifact Registry repository if it doesn't exist.
        
        Args:
            repository_name: Repository name
            repository_format: Repository format
            
        Returns:
            dict: Repository information
        """
        logger.info(f"Creating Artifact Registry repository {repository_name}")
        
        # Check if repository exists
        result = subprocess.run([
            'gcloud', 'artifacts', 'repositories', 'describe',
            repository_name, '--location', self.region
        ], capture_output=True)
        
        if result.returncode == 0:
            logger.info(f"Repository {repository_name} already exists")
        else:
            # Create repository
            subprocess.run([
                'gcloud', 'artifacts', 'repositories', 'create',
                repository_name,
                '--repository-format', repository_format,
                '--location', self.region,
                '--description', 'Repository for RL system images'
            ], check=True)
            
            logger.info(f"Created repository {repository_name}")
        
        # Get repository path
        repo_path = f"{self.region}-docker.pkg.dev/{self.project_id}/{repository_name}"
        
        return {
            'name': repository_name,
            'path': repo_path
        }
    
    def build_and_push_image(self, 
                           repository_path: str,
                           image_name: str = 'rl-app',
                           tag: str = 'latest',
                           dockerfile_path: str = './Dockerfile',
                           build_context: str = '.') -> str:
        """Build and push Docker image to Artifact Registry.
        
        Args:
            repository_path: Repository path
            image_name: Image name
            tag: Image tag
            dockerfile_path: Path to Dockerfile
            build_context: Docker build context
            
        Returns:
            str: Full image URI
        """
        # Configure Docker to use gcloud as a credential helper
        subprocess.run([
            'gcloud', 'auth', 'configure-docker', 
            f"{self.region}-docker.pkg.dev"
        ], check=True)
        
        # Build image
        full_image_uri = f"{repository_path}/{image_name}:{tag}"
        
        logger.info(f"Building Docker image: {full_image_uri}")
        subprocess.run([
            'docker', 'build', '-t', full_image_uri, 
            '-f', dockerfile_path, build_context
        ], check=True)
        
        # Push image
        logger.info(f"Pushing Docker image: {full_image_uri}")
        subprocess.run([
            'docker', 'push', full_image_uri
        ], check=True)
        
        return full_image_uri
    
    def create_gke_cluster(self, 
                         machine_type: str = 'e2-standard-2',
                         num_nodes: int = 3) -> Dict[str, Any]:
        """Create GKE cluster if it doesn't exist.
        
        Args:
            machine_type: Machine type for nodes
            num_nodes: Number of nodes
            
        Returns:
            dict: Cluster information
        """
        logger.info(f"Creating GKE cluster: {self.cluster_name}")
        
        # Check if cluster exists
        result = subprocess.run([
            'gcloud', 'container', 'clusters', 'describe',
            self.cluster_name, '--zone', self.zone
        ], capture_output=True)
        
        if result.returncode == 0:
            logger.info(f"Cluster {self.cluster_name} already exists")
        else:
            # Create cluster
            subprocess.run([
                'gcloud', 'container', 'clusters', 'create',
                self.cluster_name,
                '--zone', self.zone,
                '--machine-type', machine_type,
                '--num-nodes', str(num_nodes),
                '--enable-autoscaling',
                '--min-nodes', '1',
                '--max-nodes', '5'
            ], check=True)
            
            logger.info(f"Created cluster {self.cluster_name}")
        
        # Get cluster credentials
        subprocess.run([
            'gcloud', 'container', 'clusters', 'get-credentials',
            self.cluster_name, '--zone', self.zone
        ], check=True)
        
        return {
            'name': self.cluster_name,
            'zone': self.zone
        }
    
    def deploy_mongodb(self, namespace: str = 'default') -> Dict[str, Any]:
        """Deploy MongoDB using Helm.
        
        Args:
            namespace: Kubernetes namespace
            
        Returns:
            dict: MongoDB service information
        """
        logger.info("Deploying MongoDB")
        
        # Add Helm repository
        subprocess.run([
            'helm', 'repo', 'add', 'bitnami', 
            'https://charts.bitnami.com/bitnami'
        ], check=True)
        
        # Update Helm repositories
        subprocess.run([
            'helm', 'repo', 'update'
        ], check=True)
        
        # Check if MongoDB is already deployed
        result = subprocess.run([
            'helm', 'list', '-n', namespace, '--output', 'json'
        ], capture_output=True, check=True)
        
        releases = json.loads(result.stdout.decode('utf-8'))
        mongodb_exists = any(release['name'] == 'mongodb' for release in releases)
        
        if mongodb_exists:
            logger.info("MongoDB is already deployed")
        else:
            # Deploy MongoDB
            subprocess.run([
                'helm', 'install', 'mongodb', 'bitnami/mongodb',
                '--namespace', namespace,
                '--set', 'auth.enabled=true',
                '--set', 'auth.rootPassword=rlsecret',
                '--set', 'auth.username=rluser',
                '--set', 'auth.password=rlpass',
                '--set', 'auth.database=rl_results'
            ], check=True)
            
            logger.info("MongoDB deployed successfully")
        
        # Get MongoDB service details
        result = subprocess.run([
            'kubectl', 'get', 'service', 'mongodb', 
            '-n', namespace, '-o', 'json'
        ], capture_output=True, check=True)
        
        service_info = json.loads(result.stdout.decode('utf-8'))
        
        return {
            'name': 'mongodb',
            'service': service_info
        }
    
    def deploy_redis(self, namespace: str = 'default') -> Dict[str, Any]:
        """Deploy Redis using Helm.
        
        Args:
            namespace: Kubernetes namespace
            
        Returns:
            dict: Redis service information
        """
        logger.info("Deploying Redis")
        
        # Add Helm repository (if not already added)
        subprocess.run([
            'helm', 'repo', 'add', 'bitnami', 
            'https://charts.bitnami.com/bitnami'
        ], check=True)
        
        # Update Helm repositories
        subprocess.run([
            'helm', 'repo', 'update'
        ], check=True)
        
        # Check if Redis is already deployed
        result = subprocess.run([
            'helm', 'list', '-n', namespace, '--output', 'json'
        ], capture_output=True, check=True)
        
        releases = json.loads(result.stdout.decode('utf-8'))
        redis_exists = any(release['name'] == 'redis' for release in releases)
        
        if redis_exists:
            logger.info("Redis is already deployed")
        else:
            # Deploy Redis
            subprocess.run([
                'helm', 'install', 'redis', 'bitnami/redis',
                '--namespace', namespace,
                '--set', 'auth.password=rlsecret'
            ], check=True)
            
            logger.info("Redis deployed successfully")
        
        # Get Redis service details
        result = subprocess.run([
            'kubectl', 'get', 'service', 'redis-master', 
            '-n', namespace, '-o', 'json'
        ], capture_output=True, check=True)
        
        service_info = json.loads(result.stdout.decode('utf-8'))
        
        return {
            'name': 'redis',
            'service': service_info
        }
    
    def deploy_application(self, 
                         image_uri: str, 
                         namespace: str = 'default',
                         deployment_name: str = 'rl-app',
                         replicas: int = 2,
                         port: int = 5000) -> Dict[str, Any]:
        """Deploy application to GKE.
        
        Args:
            image_uri: Docker image URI
            namespace: Kubernetes namespace
            deployment_name: Deployment name
            replicas: Number of replicas
            port: Container port
            
        Returns:
            dict: Deployment and service information
        """
        logger.info(f"Deploying application: {deployment_name}")
        
        # Create deployment YAML
        deployment_yaml = f"""
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {deployment_name}
  namespace: {namespace}
spec:
  replicas: {replicas}
  selector:
    matchLabels:
      app: {deployment_name}
  template:
    metadata:
      labels:
        app: {deployment_name}
    spec:
      containers:
      - name: {deployment_name}
        image: {image_uri}
        ports:
        - containerPort: {port}
        env:
        - name: MONGODB_URI
          value: mongodb://rluser:rlpass@mongodb.{namespace}.svc.cluster.local:27017/rl_results
        - name: REDIS_URL
          value: redis://redis-master.{namespace}.svc.cluster.local:6379
        - name: ENV
          value: production
---
apiVersion: v1
kind: Service
metadata:
  name: {deployment_name}
  namespace: {namespace}
spec:
  selector:
    app: {deployment_name}
  ports:
  - port: {port}
    targetPort: {port}
  type: LoadBalancer
"""
        
        # Write deployment YAML to file
        with open('deployment.yaml', 'w') as f:
            f.write(deployment_yaml)
        
        # Apply deployment
        subprocess.run([
            'kubectl', 'apply', '-f', 'deployment.yaml'
        ], check=True)
        
        logger.info(f"Application {deployment_name} deployed")
        
        # Wait for deployment to be ready
        logger.info("Waiting for deployment to be ready")
        subprocess.run([
            'kubectl', 'rollout', 'status', 
            f"deployment/{deployment_name}", 
            '-n', namespace
        ], check=True)
        
        # Get deployment details
        result = subprocess.run([
            'kubectl', 'get', 'deployment', deployment_name, 
            '-n', namespace, '-o', 'json'
        ], capture_output=True, check=True)
        
        deployment_info = json.loads(result.stdout.decode('utf-8'))
        
        # Get service details
        result = subprocess.run([
            'kubectl', 'get', 'service', deployment_name, 
            '-n', namespace, '-o', 'json'
        ], capture_output=True, check=True)
        
        service_info = json.loads(result.stdout.decode('utf-8'))
        
        # Clean up
        os.remove('deployment.yaml')
        
        return {
            'deployment': deployment_info,
            'service': service_info
        }
    
    def deploy(self, 
             image_name: str = 'rl-app',
             tag: str = 'latest',
             replicas: int = 2) -> Dict[str, Any]:
        """Deploy the entire stack.
        
        Args:
            image_name: Docker image name
            tag: Docker image tag
            replicas: Number of application replicas
            
        Returns:
            dict: Deployment information
        """
        # Create Artifact Registry
        repository = self.create_artifact_registry()
        
        # Build and push image
        image_uri = self.build_and_push_image(
            repository['path'],
            image_name=image_name,
            tag=tag
        )
        
        # Create GKE cluster
        cluster = self.create_gke_cluster()
        
        # Deploy MongoDB
        mongodb = self.deploy_mongodb()
        
        # Deploy Redis
        redis = self.deploy_redis()
        
        # Deploy application
        app = self.deploy_application(
            image_uri=image_uri,
            replicas=replicas
        )
        
        # Get application URL
        external_ip = None
        for attempt in range(10):
            result = subprocess.run([
                'kubectl', 'get', 'service', 'rl-app', 
                '-o', 'jsonpath={.status.loadBalancer.ingress[0].ip}'
            ], capture_output=True)
            
            if result.returncode == 0 and result.stdout:
                external_ip = result.stdout.decode('utf-8')
                break
                
            logger.info("Waiting for external IP...")
            time.sleep(10)
            
        app_url = f"http://{external_ip}:5000" if external_ip else "Not available yet"
        
        return {
            'repository': repository,
            'image_uri': image_uri,
            'cluster': cluster,
            'mongodb': mongodb,
            'redis': redis,
            'application': app,
            'app_url': app_url
        }


def main():
    parser = argparse.ArgumentParser(description='Deploy RL system to GCP')
    parser.add_argument('--project-id', required=True, help='GCP project ID')
    parser.add_argument('--region', default='us-central1', help='GCP region')
    parser.add_argument('--zone', default='us-central1-a', help='GCP zone')
    parser.add_argument('--cluster-name', default='rl-cluster', help='GKE cluster name')
    parser.add_argument('--image-name', default='rl-app', help='Docker image name')
    parser.add_argument('--tag', default='latest', help='Docker image tag')
    parser.add_argument('--replicas', type=int, default=2, help='Number of application replicas')
    
    args = parser.parse_args()
    
    deployer = GCPDeployer(
        project_id=args.project_id,
        region=args.region,
        zone=args.zone,
        cluster_name=args.cluster_name
    )
    
    result = deployer.deploy(
        image_name=args.image_name,
        tag=args.tag,
        replicas=args.replicas
    )
    
    logger.info(f"Deployment complete!")
    logger.info(f"Application URL: {result['app_url']}")


if __name__ == '__main__':
    main() 
#!/usr/bin/env python3
import os
import argparse
import json
import boto3
import time
import logging
from typing import Dict, Any, List, Optional
from botocore.exceptions import ClientError

logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AWSDeployer:
    """Utility for deploying the RL system on AWS."""
    
    def __init__(self, 
               region: str = 'us-east-1',
               profile: Optional[str] = None,
               cluster_name: str = 'rl-cluster',
               task_definition: str = 'rl-task',
               service_name: str = 'rl-service'):
        """Initialize AWS deployer.
        
        Args:
            region: AWS region
            profile: AWS profile name (optional)
            cluster_name: ECS cluster name
            task_definition: ECS task definition name
            service_name: ECS service name
        """
        self.region = region
        self.cluster_name = cluster_name
        self.task_definition = task_definition
        self.service_name = service_name
        
        # Initialize AWS session
        if profile:
            self.session = boto3.Session(profile_name=profile, region_name=region)
        else:
            self.session = boto3.Session(region_name=region)
        
        # Initialize AWS clients
        self.ec2 = self.session.client('ec2')
        self.ecr = self.session.client('ecr')
        self.ecs = self.session.client('ecs')
        self.cloudformation = self.session.client('cloudformation')
        
    def create_ecr_repository(self, repo_name: str) -> Dict[str, Any]:
        """Create ECR repository if it doesn't exist.
        
        Args:
            repo_name: Repository name
            
        Returns:
            dict: Repository information
        """
        try:
            # Check if repository exists
            response = self.ecr.describe_repositories(repositoryNames=[repo_name])
            logger.info(f"Repository {repo_name} already exists")
            return response['repositories'][0]
        except ClientError as e:
            if e.response['Error']['Code'] == 'RepositoryNotFoundException':
                # Create repository
                logger.info(f"Creating repository {repo_name}")
                response = self.ecr.create_repository(
                    repositoryName=repo_name,
                    imageTagMutability='MUTABLE',
                    imageScanningConfiguration={'scanOnPush': True}
                )
                return response['repository']
            else:
                raise
    
    def get_ecr_login_command(self) -> str:
        """Get ECR login command.
        
        Returns:
            str: Docker login command
        """
        response = self.ecr.get_authorization_token()
        auth_data = response['authorizationData'][0]
        token = auth_data['authorizationToken']
        endpoint = auth_data['proxyEndpoint']
        
        return f"aws ecr get-login-password --region {self.region} | docker login --username AWS --password-stdin {endpoint}"
    
    def build_and_push_image(self, 
                           repo_name: str, 
                           tag: str = 'latest',
                           dockerfile_path: str = './Dockerfile',
                           build_context: str = '.') -> str:
        """Build and push Docker image to ECR.
        
        Args:
            repo_name: Repository name
            tag: Image tag
            dockerfile_path: Path to Dockerfile
            build_context: Docker build context
            
        Returns:
            str: Full image URI
        """
        # Create ECR repository
        repo = self.create_ecr_repository(repo_name)
        repo_uri = repo['repositoryUri']
        
        # Build image
        image_uri = f"{repo_uri}:{tag}"
        build_cmd = f"docker build -t {image_uri} -f {dockerfile_path} {build_context}"
        logger.info(f"Building image: {build_cmd}")
        
        # For demonstration purposes - in actual code we would execute these commands
        logger.info(f"Build command: {build_cmd}")
        
        # Push image
        push_cmd = f"docker push {image_uri}"
        logger.info(f"Push command: {push_cmd}")
        
        return image_uri
    
    def create_ecs_cluster(self) -> Dict[str, Any]:
        """Create ECS cluster if it doesn't exist.
        
        Returns:
            dict: Cluster information
        """
        try:
            # Check if cluster exists
            response = self.ecs.describe_clusters(clusters=[self.cluster_name])
            
            if response['clusters'] and response['clusters'][0]['status'] != 'INACTIVE':
                logger.info(f"Cluster {self.cluster_name} already exists")
                return response['clusters'][0]
            
            # Create cluster
            logger.info(f"Creating cluster {self.cluster_name}")
            response = self.ecs.create_cluster(
                clusterName=self.cluster_name,
                capacityProviders=['FARGATE'],
                defaultCapacityProviderStrategy=[
                    {
                        'capacityProvider': 'FARGATE',
                        'weight': 1
                    }
                ],
                settings=[
                    {
                        'name': 'containerInsights',
                        'value': 'enabled'
                    }
                ]
            )
            return response['cluster']
        except ClientError as e:
            logger.error(f"Error creating ECS cluster: {e}")
            raise
    
    def create_or_update_task_definition(self, 
                                       image_uri: str,
                                       cpu: str = '1024',
                                       memory: str = '2048') -> Dict[str, Any]:
        """Create or update ECS task definition.
        
        Args:
            image_uri: ECR image URI
            cpu: CPU units
            memory: Memory in MiB
            
        Returns:
            dict: Task definition information
        """
        try:
            # Define container definitions
            container_definitions = [
                {
                    'name': 'rl-app',
                    'image': image_uri,
                    'essential': True,
                    'portMappings': [
                        {
                            'containerPort': 5000,
                            'hostPort': 5000,
                            'protocol': 'tcp'
                        }
                    ],
                    'environment': [
                        {
                            'name': 'MONGODB_URI',
                            'value': os.environ.get('MONGODB_URI', 'mongodb://mongo:27017/rl_results')
                        },
                        {
                            'name': 'REDIS_URL',
                            'value': os.environ.get('REDIS_URL', 'redis://redis:6379/0')
                        },
                        {
                            'name': 'ENV',
                            'value': 'production'
                        }
                    ],
                    'logConfiguration': {
                        'logDriver': 'awslogs',
                        'options': {
                            'awslogs-group': f'/ecs/{self.task_definition}',
                            'awslogs-region': self.region,
                            'awslogs-stream-prefix': 'ecs'
                        }
                    }
                }
            ]
            
            # Create task definition
            logger.info(f"Creating task definition {self.task_definition}")
            response = self.ecs.register_task_definition(
                family=self.task_definition,
                executionRoleArn=os.environ.get('ECS_EXECUTION_ROLE_ARN', 'arn:aws:iam::123456789012:role/ecsTaskExecutionRole'),
                taskRoleArn=os.environ.get('ECS_TASK_ROLE_ARN', 'arn:aws:iam::123456789012:role/ecsTaskRole'),
                networkMode='awsvpc',
                containerDefinitions=container_definitions,
                requiresCompatibilities=['FARGATE'],
                cpu=cpu,
                memory=memory
            )
            
            return response['taskDefinition']
        except ClientError as e:
            logger.error(f"Error creating task definition: {e}")
            raise
    
    def create_or_update_service(self, 
                               subnet_ids: List[str],
                               security_group_ids: List[str],
                               desired_count: int = 1) -> Dict[str, Any]:
        """Create or update ECS service.
        
        Args:
            subnet_ids: Subnet IDs for the service
            security_group_ids: Security group IDs
            desired_count: Desired task count
            
        Returns:
            dict: Service information
        """
        try:
            # Check if service exists
            try:
                response = self.ecs.describe_services(
                    cluster=self.cluster_name,
                    services=[self.service_name]
                )
                
                if response['services'] and response['services'][0]['status'] != 'INACTIVE':
                    # Update service
                    logger.info(f"Updating service {self.service_name}")
                    response = self.ecs.update_service(
                        cluster=self.cluster_name,
                        service=self.service_name,
                        desiredCount=desired_count,
                        taskDefinition=self.task_definition
                    )
                    return response['service']
            except:
                pass
            
            # Create service
            logger.info(f"Creating service {self.service_name}")
            response = self.ecs.create_service(
                cluster=self.cluster_name,
                serviceName=self.service_name,
                taskDefinition=self.task_definition,
                desiredCount=desired_count,
                launchType='FARGATE',
                networkConfiguration={
                    'awsvpcConfiguration': {
                        'subnets': subnet_ids,
                        'securityGroups': security_group_ids,
                        'assignPublicIp': 'ENABLED'
                    }
                }
            )
            
            return response['service']
        except ClientError as e:
            logger.error(f"Error creating/updating service: {e}")
            raise
    
    def deploy(self, 
             repo_name: str,
             subnet_ids: List[str],
             security_group_ids: List[str],
             tag: str = 'latest',
             desired_count: int = 1) -> Dict[str, Any]:
        """Deploy the application to ECS.
        
        Args:
            repo_name: ECR repository name
            subnet_ids: Subnet IDs for the service
            security_group_ids: Security group IDs
            tag: Image tag
            desired_count: Desired task count
            
        Returns:
            dict: Deployment information
        """
        # Build and push image
        image_uri = self.build_and_push_image(repo_name, tag)
        
        # Create ECS cluster
        cluster = self.create_ecs_cluster()
        
        # Create task definition
        task_def = self.create_or_update_task_definition(image_uri)
        
        # Create or update service
        service = self.create_or_update_service(subnet_ids, security_group_ids, desired_count)
        
        return {
            'image_uri': image_uri,
            'cluster': cluster,
            'task_definition': task_def,
            'service': service
        }
    
    def create_cloudformation_stack(self, 
                                  stack_name: str,
                                  template_path: str,
                                  parameters: Dict[str, str] = None) -> Dict[str, Any]:
        """Create or update CloudFormation stack.
        
        Args:
            stack_name: Stack name
            template_path: Path to CloudFormation template
            parameters: Stack parameters
            
        Returns:
            dict: Stack information
        """
        try:
            # Read template
            with open(template_path, 'r') as f:
                template_body = f.read()
                
            # Convert parameters
            cf_parameters = []
            if parameters:
                for key, value in parameters.items():
                    cf_parameters.append({
                        'ParameterKey': key,
                        'ParameterValue': value
                    })
            
            # Check if stack exists
            try:
                self.cloudformation.describe_stacks(StackName=stack_name)
                
                # Update stack
                logger.info(f"Updating stack {stack_name}")
                response = self.cloudformation.update_stack(
                    StackName=stack_name,
                    TemplateBody=template_body,
                    Parameters=cf_parameters,
                    Capabilities=['CAPABILITY_IAM', 'CAPABILITY_NAMED_IAM']
                )
                
                # Wait for stack update to complete
                waiter = self.cloudformation.get_waiter('stack_update_complete')
                waiter.wait(StackName=stack_name)
            except ClientError as e:
                if 'does not exist' in str(e):
                    # Create stack
                    logger.info(f"Creating stack {stack_name}")
                    response = self.cloudformation.create_stack(
                        StackName=stack_name,
                        TemplateBody=template_body,
                        Parameters=cf_parameters,
                        Capabilities=['CAPABILITY_IAM', 'CAPABILITY_NAMED_IAM']
                    )
                    
                    # Wait for stack creation to complete
                    waiter = self.cloudformation.get_waiter('stack_create_complete')
                    waiter.wait(StackName=stack_name)
                elif 'No updates are to be performed' in str(e):
                    logger.info(f"No updates needed for stack {stack_name}")
                    return self.cloudformation.describe_stacks(StackName=stack_name)['Stacks'][0]
                else:
                    raise
            
            # Get stack info
            response = self.cloudformation.describe_stacks(StackName=stack_name)
            return response['Stacks'][0]
        except ClientError as e:
            logger.error(f"Error with CloudFormation stack: {e}")
            raise
    
    def deploy_infrastructure(self, 
                            stack_name: str = 'rl-infrastructure',
                            template_path: str = './deployment/cloudformation/infrastructure.yaml') -> Dict[str, Any]:
        """Deploy infrastructure using CloudFormation.
        
        Args:
            stack_name: Stack name
            template_path: Path to CloudFormation template
            
        Returns:
            dict: Infrastructure stack information
        """
        # Define parameters
        parameters = {
            'ClusterName': self.cluster_name,
            'ServiceName': self.service_name
        }
        
        # Create or update stack
        stack = self.create_cloudformation_stack(stack_name, template_path, parameters)
        
        # Get outputs
        outputs = {output['OutputKey']: output['OutputValue'] for output in stack['Outputs']}
        
        return {
            'stack': stack,
            'outputs': outputs
        }


def main():
    parser = argparse.ArgumentParser(description='Deploy RL system to AWS')
    parser.add_argument('--region', default='us-east-1', help='AWS region')
    parser.add_argument('--profile', help='AWS profile name')
    parser.add_argument('--repo-name', default='rl-system', help='ECR repository name')
    parser.add_argument('--subnet-ids', help='Comma-separated list of subnet IDs')
    parser.add_argument('--security-group-ids', help='Comma-separated list of security group IDs')
    parser.add_argument('--tag', default='latest', help='Image tag')
    parser.add_argument('--stack-name', default='rl-infrastructure', help='CloudFormation stack name')
    parser.add_argument('--template-path', default='./deployment/cloudformation/infrastructure.yaml', 
                       help='Path to CloudFormation template')
    parser.add_argument('--desired-count', type=int, default=1, help='Desired task count')
    
    args = parser.parse_args()
    
    # Convert subnet and security group IDs to lists
    subnet_ids = args.subnet_ids.split(',') if args.subnet_ids else []
    security_group_ids = args.security_group_ids.split(',') if args.security_group_ids else []
    
    deployer = AWSDeployer(
        region=args.region,
        profile=args.profile
    )
    
    if os.path.exists(args.template_path):
        # Deploy infrastructure using CloudFormation
        logger.info("Deploying infrastructure...")
        infra = deployer.deploy_infrastructure(args.stack_name, args.template_path)
        
        # Get subnet and security group IDs from stack outputs if not provided
        if not subnet_ids and 'SubnetIds' in infra['outputs']:
            subnet_ids = infra['outputs']['SubnetIds'].split(',')
        if not security_group_ids and 'SecurityGroupIds' in infra['outputs']:
            security_group_ids = infra['outputs']['SecurityGroupIds'].split(',')
    
    if subnet_ids and security_group_ids:
        # Deploy application
        logger.info("Deploying application...")
        result = deployer.deploy(
            repo_name=args.repo_name,
            subnet_ids=subnet_ids,
            security_group_ids=security_group_ids,
            tag=args.tag,
            desired_count=args.desired_count
        )
        
        logger.info(f"Deployment complete! Service URL: {result.get('service', {}).get('serviceUrl', 'Not available')}")
    else:
        logger.error("Subnet IDs and security group IDs are required for deployment.")
        parser.print_help()
        

if __name__ == '__main__':
    main() 
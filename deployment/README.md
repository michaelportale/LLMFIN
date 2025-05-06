# RL System Deployment

This directory contains the necessary files and scripts to deploy the reinforcement learning system to production environments.

## Quick Start

### Local Deployment

To deploy the system locally using Docker Compose:

```bash
# Build and start all containers
docker-compose up -d

# Check the status of containers
docker-compose ps

# View logs
docker-compose logs -f

# Stop the system
docker-compose down
```

### AWS Deployment

To deploy to AWS:

```bash
# Make sure you have AWS CLI configured
aws configure

# Run the deployment script
python aws_deploy.py --region us-east-1 --stack-name rl-infrastructure
```

### GCP Deployment

To deploy to Google Cloud Platform:

```bash
# Make sure you've installed and configured gcloud CLI
gcloud auth login
gcloud config set project <YOUR_PROJECT_ID>

# Run the deployment script
python gcp_deploy.py --project-id <YOUR_PROJECT_ID> --region us-central1
```

## Infrastructure

The deployment setup includes:

- **Application**: Containerized RL application
- **MongoDB**: For storing model metadata and results
- **Redis**: For caching and job queuing
- **Prometheus/Grafana**: For monitoring

## Directory Structure

- `aws_deploy.py`: AWS deployment script
- `gcp_deploy.py`: GCP deployment script
- `cloudformation/`: CloudFormation templates for AWS infrastructure
- `monitoring/`: Prometheus and Grafana configuration

## Environment Variables

The following environment variables are used:

- `MONGODB_URI`: URI for MongoDB connection
- `REDIS_URL`: URL for Redis connection
- `ENV`: Environment (development, production)

## Monitoring

Access Grafana at: `http://<host>:3000`
- Default username: admin
- Default password: admin

## Automated Testing

Run tests with:

```bash
python -m pytest ../tests/
```

## CI/CD Pipeline

The system uses GitHub Actions for continuous integration and deployment:

1. Run tests on each commit
2. Build and push Docker image
3. Deploy to development environment on successful build
4. Manual approval for production deployment 
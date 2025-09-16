# Production Deployment Guide for WhatsApp RAG

This guide provides comprehensive instructions for deploying the WhatsApp RAG system in production environments with different deployment strategies.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Docker Deployment](#docker-deployment)
3. [AWS Cloud Deployment](#aws-cloud-deployment)
4. [Configuration Management](#configuration-management)
5. [Monitoring and Observability](#monitoring-and-observability)
6. [Security Best Practices](#security-best-practices)
7. [Troubleshooting](#troubleshooting)

## Prerequisites

### System Requirements

**Minimum Requirements:**
- CPU: 2 cores
- RAM: 4GB
- Storage: 10GB free space
- OS: Linux (Ubuntu 20.04+), macOS, or Windows with Docker

**Recommended for Production:**
- CPU: 4+ cores
- RAM: 8GB+
- Storage: 50GB+ SSD
- Network: Stable internet connection

### Required Software

- Docker and Docker Compose
- AWS CLI (for AWS deployment)
- Git
- curl/wget

### API Access

- GitHub token with appropriate scopes for API access
- AWS account with necessary permissions (for AWS deployment)

## Docker Deployment

### 1. Quick Start with Docker Compose

```bash
# Clone and navigate to examples
cd examples/docker

# Copy environment template
cp ../../.env.example .env

# Edit configuration
nano .env  # Add your GitHub token

# Deploy development environment
docker-compose up -d

# Or deploy production environment
docker-compose -f docker-compose.prod.yml up -d
```

### 2. Production Docker Deployment

For production deployments with full monitoring stack:

```bash
# Deploy with monitoring (Prometheus + Grafana)
docker-compose -f docker-compose.prod.yml --profile monitoring --profile cache up -d

# Access services
echo "Application: http://localhost"
echo "Grafana: http://localhost:3000"
echo "Prometheus: http://localhost:9090"
```

### 3. Using Deployment Script

```bash
# Make script executable (Linux/macOS)
chmod +x deploy.sh

# Deploy development environment
./deploy.sh deploy-dev

# Deploy production environment
./deploy.sh deploy-prod

# Deploy with full monitoring
./deploy.sh deploy-monitor

# Check status
./deploy.sh status

# View logs
./deploy.sh logs

# Stop services
./deploy.sh stop

# Clean up everything
./deploy.sh destroy
```

### 4. Advanced Docker Configuration

#### Resource Limits

```yaml
# In docker-compose.prod.yml
services:
  whatsapp-rag:
    deploy:
      resources:
        limits:
          cpus: '4.0'
          memory: 4G
        reservations:
          cpus: '2.0'
          memory: 2G
```

#### Volume Management

```yaml
volumes:
  # Persistent data storage
  whatsapp_data:
    driver: local
    driver_opts:
      type: none
      o: bind
      device: /opt/whatsapp-rag/data
  
  # Application logs
  app_logs:
    driver: local
```

## AWS Cloud Deployment

### 1. Prerequisites for AWS

```bash
# Install AWS CLI
pip install awscli

# Configure credentials
aws configure

# Verify access
aws sts get-caller-identity
```

### 2. Quick AWS Deployment

```bash
cd examples/aws

# Deploy everything automatically
./deploy.sh deploy

# Check deployment status
./deploy.sh status

# View logs
./deploy.sh logs

# Scale service
./deploy.sh scale 4

# Update application
./deploy.sh update

# Clean up
./deploy.sh destroy
```

### 3. Manual AWS Deployment Steps

#### Step 1: Create ECR Repository

```bash
aws ecr create-repository --repository-name whatsapp-rag --region us-east-1
```

#### Step 2: Build and Push Docker Image

```bash
# Get ECR URI
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_URI="${ACCOUNT_ID}.dkr.ecr.us-east-1.amazonaws.com/whatsapp-rag"

# Login to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin $ECR_URI

# Build and push
docker build -f ../docker/production.Dockerfile -t whatsapp-rag ../../
docker tag whatsapp-rag:latest $ECR_URI:latest
docker push $ECR_URI:latest
```

#### Step 3: Deploy CloudFormation Stack

```bash
aws cloudformation deploy \
  --template-file cloudformation-template.yaml \
  --stack-name whatsapp-rag \
  --parameter-overrides GitHubToken=your_token_here \
  --capabilities CAPABILITY_IAM \
  --region us-east-1
```

### 4. AWS Architecture Overview

The CloudFormation template creates:

- **ECS Fargate Cluster**: Serverless containers
- **Application Load Balancer**: Traffic distribution and SSL termination
- **VPC with Public/Private Subnets**: Network isolation
- **S3 Bucket**: Data storage with versioning
- **Secrets Manager**: Secure token storage
- **CloudWatch**: Logging and monitoring
- **Auto Scaling**: Automatic scaling based on CPU usage
- **IAM Roles**: Least privilege access

### 5. Cost Optimization

```bash
# Use Fargate Spot for non-critical workloads
aws cloudformation deploy \
  --template-file cloudformation-template.yaml \
  --stack-name whatsapp-rag \
  --parameter-overrides \
    GitHubToken=your_token \
    InstanceType=FARGATE_SPOT \
  --capabilities CAPABILITY_IAM
```

**Estimated Monthly Costs (us-east-1):**
- 2x Fargate tasks (1 vCPU, 2GB): ~$70
- Application Load Balancer: ~$20
- S3 storage (typical usage): <$5
- CloudWatch logs: <$10
- **Total**: ~$100-120/month

## Configuration Management

### 1. Environment-Specific Configuration

#### Development Configuration

```bash
# Copy development template
cp examples/config/development.env .env

# Key settings for development:
# - DEBUG=true
# - LOG_LEVEL=DEBUG
# - No authentication required
# - Relaxed rate limits
```

#### Production Configuration

```bash
# Copy production template
cp examples/config/production.env .env

# Key settings for production:
# - GRADIO_USERNAME and GRADIO_PASSWORD required
# - LOG_LEVEL=WARNING
# - Strict rate limits
# - Monitoring enabled
```

### 2. RAG System Configuration

```yaml
# examples/config/rag_config.yaml
embedding:
  model_name: "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
  batch_size: 32

rag:
  top_k: 5
  similarity_threshold: 0.7
  use_mmr: true

performance:
  max_workers: 4
  chunk_processing_batch_size: 100
```

### 3. Loading Configuration in Code

```python
import yaml
import os
from pathlib import Path

def load_config(config_path="examples/config/rag_config.yaml"):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Override with environment-specific config
    env = os.getenv("ENVIRONMENT", "development")
    if env in config:
        for section, values in config[env].items():
            config[section].update(values)
    
    return config

# Usage
config = load_config()
embedding_model = config['embedding']['model_name']
```

## Monitoring and Observability

### 1. Docker Monitoring Stack

The production Docker Compose includes:

- **Prometheus**: Metrics collection
- **Grafana**: Dashboards and visualization
- **Redis**: Caching and session storage

```bash
# Deploy with monitoring
docker-compose -f docker-compose.prod.yml --profile monitoring up -d

# Access Grafana
open http://localhost:3000
# Default credentials: admin/admin
```

### 2. AWS CloudWatch Integration

AWS deployment automatically configures:

- **Application logs** in CloudWatch Logs
- **ECS metrics** for CPU, memory, network
- **Load balancer metrics** for request count, latency
- **Custom application metrics** via CloudWatch API

### 3. Health Checks

#### Docker Health Checks

```dockerfile
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:7860/ || exit 1
```

#### AWS Health Checks

```yaml
HealthCheckPath: /
HealthCheckIntervalSeconds: 30
HealthyThresholdCount: 2
UnhealthyThresholdCount: 5
```

### 4. Application Metrics

Enable metrics collection:

```bash
# Environment variable
ENABLE_METRICS=true
METRICS_PORT=9090

# Access metrics endpoint
curl http://localhost:9090/metrics
```

Key metrics to monitor:
- Request latency (p50, p95, p99)
- Error rate
- Memory usage
- Embedding processing time
- Active sessions

## Security Best Practices

### 1. Authentication and Authorization

#### Basic Authentication (Gradio)

```bash
# Production environment
GRADIO_USERNAME=admin
GRADIO_PASSWORD=secure_complex_password_123!
GRADIO_AUTH_MESSAGE="Acceso restringido - Solo personal autorizado"
```

#### Advanced Authentication

For enterprise deployments, consider:
- OAuth2/SAML integration
- LDAP authentication
- Multi-factor authentication
- Role-based access control

### 2. Network Security

#### Docker Network Isolation

```yaml
networks:
  whatsapp-rag-network:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.0.0/16
```

#### AWS Security Groups

```yaml
SecurityGroupIngress:
  - IpProtocol: tcp
    FromPort: 443
    ToPort: 443
    CidrIp: 0.0.0.0/0  # Restrict to specific IPs in production
  - IpProtocol: tcp
    FromPort: 7860
    ToPort: 7860
    SourceSecurityGroupId: !Ref LoadBalancerSecurityGroup
```

### 3. Data Protection

#### Encryption at Rest

```bash
# S3 bucket encryption
BucketEncryption:
  ServerSideEncryptionConfiguration:
    - ServerSideEncryptionByDefault:
        SSEAlgorithm: AES256
```

#### Encryption in Transit

```yaml
# HTTPS/TLS configuration
ssl_protocols: TLSv1.2 TLSv1.3
ssl_ciphers: ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256
```

### 4. Secrets Management

#### Docker Secrets

```yaml
secrets:
  github_token:
    external: true
services:
  whatsapp-rag:
    secrets:
      - github_token
```

#### AWS Secrets Manager

```yaml
GitHubTokenSecret:
  Type: AWS::SecretsManager::Secret
  Properties:
    Name: !Sub '${AWS::StackName}/github-token'
    SecretString: !Sub |
      {
        "token": "${GitHubToken}"
      }
```

### 5. Rate Limiting and DDoS Protection

```bash
# Application-level rate limiting
MAX_REQUESTS_PER_MINUTE=30
MAX_CONCURRENT_REQUESTS=3

# File upload limits
MAX_FILE_SIZE_MB=10
```

#### Nginx Rate Limiting

```nginx
limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
limit_req zone=api burst=20 nodelay;
```

## Troubleshooting

### 1. Common Issues

#### Docker Issues

**Container fails to start:**
```bash
# Check logs
docker logs whatsapp-rag

# Check health status
docker inspect --format='{{json .State.Health}}' whatsapp-rag
```

**Out of memory:**
```bash
# Check memory usage
docker stats whatsapp-rag

# Increase memory limits
docker-compose up -d --scale whatsapp-rag=1 --memory=4g
```

**Permission denied:**
```bash
# Fix file permissions
sudo chown -R $USER:$USER ./data ./logs

# Check Docker socket permissions
sudo usermod -aG docker $USER
```

#### AWS Issues

**ECS tasks not starting:**
```bash
# Check ECS service events
aws ecs describe-services --cluster whatsapp-rag-cluster --services whatsapp-rag-service

# Check task logs
aws logs tail /ecs/whatsapp-rag --follow
```

**Load balancer health check failures:**
```bash
# Check target group health
aws elbv2 describe-target-health --target-group-arn YOUR_TARGET_GROUP_ARN

# Verify application is responding
curl -f http://YOUR_ALB_DNS_NAME/
```

**High costs:**
```bash
# Check current costs
aws ce get-cost-and-usage --time-period Start=2024-01-01,End=2024-01-31 --granularity MONTHLY --metrics BlendedCost

# Use Spot instances for cost savings
aws cloudformation deploy --parameter-overrides InstanceType=FARGATE_SPOT
```

### 2. Application Issues

#### GitHub API Rate Limits

```bash
# Check rate limit status
curl -H "Authorization: token YOUR_TOKEN" https://api.github.com/rate_limit

# Use different tokens for different services
GITHUB_TOKEN_PRIMARY=token1
GITHUB_TOKEN_SECONDARY=token2
```

#### Embedding Model Issues

```bash
# Clear model cache
rm -rf ~/.cache/torch/sentence_transformers/

# Use smaller model for resource constraints
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

#### Memory Issues

```bash
# Monitor memory usage
watch -n 1 'free -h && echo "---" && ps aux --sort=-%mem | head -10'

# Optimize chunking strategy
CHUNK_SIZE=500
CHUNK_PROCESSING_BATCH_SIZE=50
EMBEDDING_BATCH_SIZE=16
```

### 3. Performance Optimization

#### Database Query Optimization

```python
# Use connection pooling
DATABASE_POOL_SIZE=10
DATABASE_MAX_OVERFLOW=20

# Enable query optimization
SQLALCHEMY_ECHO=false
SQLALCHEMY_ENGINE_OPTIONS={'pool_pre_ping': True}
```

#### Cache Optimization

```bash
# Redis cache configuration
REDIS_HOST=redis
REDIS_PORT=6379
CACHE_TTL=3600
CACHE_MAX_SIZE=1000

# Memory cache for development
CACHE_TYPE=memory
```

#### Network Optimization

```nginx
# Enable compression
gzip on;
gzip_vary on;
gzip_comp_level 6;

# Enable HTTP/2
listen 443 ssl http2;

# Optimize keepalive
keepalive_timeout 65;
keepalive_requests 100;
```

### 4. Backup and Recovery

#### Automated Backups

```bash
# Create backup script
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="backups/$DATE"

mkdir -p "$BACKUP_DIR"

# Backup data volumes
docker run --rm -v whatsapp-rag_app_logs:/data -v $(pwd)/$BACKUP_DIR:/backup alpine \
    tar czf /backup/logs.tar.gz -C /data .

# Backup configuration
cp -r config "$BACKUP_DIR/"

# Upload to S3
aws s3 sync "$BACKUP_DIR" s3://your-backup-bucket/whatsapp-rag/$DATE/
```

#### Disaster Recovery

```bash
# Restore from backup
RESTORE_DATE="20240115_143000"

# Download from S3
aws s3 sync s3://your-backup-bucket/whatsapp-rag/$RESTORE_DATE/ ./restore/

# Restore data
docker run --rm -v whatsapp-rag_app_logs:/data -v $(pwd)/restore:/backup alpine \
    tar xzf /backup/logs.tar.gz -C /data

# Restart services
docker-compose restart
```

### 5. Support and Community

For additional support:

1. **GitHub Issues**: Report bugs and feature requests
2. **Documentation**: Check the main README and API documentation
3. **Community**: Join discussions and share experiences
4. **Professional Support**: Contact for enterprise deployment assistance

## Conclusion

This guide provides a comprehensive foundation for deploying WhatsApp RAG in production environments. Choose the deployment method that best fits your infrastructure requirements, scale, and budget constraints.

Remember to:
- Always test in a staging environment first
- Monitor performance and costs regularly
- Keep security patches up to date
- Maintain regular backups
- Document any customizations for your team
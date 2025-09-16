#!/bin/bash

# AWS deployment script for WhatsApp RAG
set -e

# Configuration
STACK_NAME="whatsapp-rag"
REGION="${AWS_REGION:-us-east-1}"
TEMPLATE_FILE="cloudformation-template.yaml"
ENV_FILE="../../.env"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Logging functions
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

info() {
    echo -e "${BLUE}[INFO] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[WARNING] $1${NC}"
}

error() {
    echo -e "${RED}[ERROR] $1${NC}"
    exit 1
}

# Check dependencies
check_dependencies() {
    log "Checking dependencies..."
    
    # Check AWS CLI
    if ! command -v aws &> /dev/null; then
        error "AWS CLI not found. Please install it first: https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2.html"
    fi
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        error "Docker not found. Please install Docker first."
    fi
    
    # Check Docker daemon
    if ! docker info &> /dev/null; then
        error "Docker daemon is not running. Please start Docker."
    fi
    
    # Check AWS credentials
    if ! aws sts get-caller-identity &> /dev/null; then
        error "AWS credentials not configured. Run 'aws configure' first."
    fi
    
    # Get account info
    ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
    CALLER_IDENTITY=$(aws sts get-caller-identity --query Arn --output text)
    
    info "AWS Account: $ACCOUNT_ID"
    info "AWS Region: $REGION" 
    info "Caller Identity: $CALLER_IDENTITY"
    
    log "Dependencies OK"
}

# Load environment variables
load_env() {
    if [ -f "$ENV_FILE" ]; then
        log "Loading environment from $ENV_FILE..."
        set -a
        source "$ENV_FILE"
        set +a
    fi
}

# Get GitHub token
get_github_token() {
    load_env
    
    if [ -n "$GITHUB_TOKEN" ] && [ "$GITHUB_TOKEN" != "your_github_token_here" ]; then
        echo "$GITHUB_TOKEN"
        return
    fi
    
    if [ -n "$OPENAI_API_KEY" ] && [ "$OPENAI_API_KEY" != "your_github_token_here" ]; then
        echo "$OPENAI_API_KEY"
        return
    fi
    
    warn "GitHub token not found in $ENV_FILE"
    read -s -p "Enter your GitHub token: " token
    echo
    
    if [ -z "$token" ]; then
        error "GitHub token is required"
    fi
    
    echo "$token"
}

# Create ECR repository
setup_ecr() {
    local repo_name="$STACK_NAME-app"
    local ecr_uri="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${repo_name}"
    
    log "Setting up ECR repository..."
    
    # Create repository if it doesn't exist
    if ! aws ecr describe-repositories --repository-names "$repo_name" --region "$REGION" &> /dev/null; then
        info "Creating ECR repository: $repo_name"
        aws ecr create-repository --repository-name "$repo_name" --region "$REGION"
    else
        info "ECR repository already exists: $repo_name"
    fi
    
    # Login to ECR
    log "Logging into ECR..."
    aws ecr get-login-password --region "$REGION" | docker login --username AWS --password-stdin "$ecr_uri"
    
    echo "$ecr_uri"
}

# Build and push Docker image
build_and_push() {
    local ecr_uri="$1"
    
    log "Building Docker image..."
    
    # Build image with production Dockerfile
    docker build \
        -f ../docker/production.Dockerfile \
        -t "$STACK_NAME:latest" \
        ../..
    
    # Tag for ECR
    docker tag "$STACK_NAME:latest" "$ecr_uri:latest"
    docker tag "$STACK_NAME:latest" "$ecr_uri:$(date +%Y%m%d-%H%M%S)"
    
    log "Pushing to ECR..."
    docker push "$ecr_uri:latest"
    docker push "$ecr_uri:$(date +%Y%m%d-%H%M%S)"
    
    info "Image pushed successfully to $ecr_uri"
}

# Deploy CloudFormation stack
deploy_stack() {
    local github_token="$1"
    
    log "Deploying CloudFormation stack..."
    
    # Validate template first
    aws cloudformation validate-template \
        --template-body file://"$TEMPLATE_FILE" \
        --region "$REGION"
    
    # Deploy stack
    aws cloudformation deploy \
        --template-file "$TEMPLATE_FILE" \
        --stack-name "$STACK_NAME" \
        --parameter-overrides GitHubToken="$github_token" \
        --capabilities CAPABILITY_IAM \
        --region "$REGION" \
        --tags \
            Project=WhatsAppRAG \
            Environment=Production \
            ManagedBy=CloudFormation
    
    log "CloudFormation deployment complete"
}

# Get stack outputs
get_outputs() {
    log "Getting stack outputs..."
    
    aws cloudformation describe-stacks \
        --stack-name "$STACK_NAME" \
        --region "$REGION" \
        --query 'Stacks[0].Outputs[].[OutputKey,OutputValue]' \
        --output table
}

# Wait for service to be stable
wait_for_service() {
    log "Waiting for ECS service to stabilize..."
    
    local cluster_name
    cluster_name=$(aws cloudformation describe-stacks \
        --stack-name "$STACK_NAME" \
        --region "$REGION" \
        --query 'Stacks[0].Outputs[?OutputKey==`ECSClusterName`].OutputValue' \
        --output text)
    
    if [ -n "$cluster_name" ]; then
        aws ecs wait services-stable \
            --cluster "$cluster_name" \
            --services "${STACK_NAME}-service" \
            --region "$REGION"
        
        log "Service is stable"
    fi
}

# Upload sample data to S3
upload_sample_data() {
    log "Uploading sample data..."
    
    local bucket_name
    bucket_name=$(aws cloudformation describe-stacks \
        --stack-name "$STACK_NAME" \
        --region "$REGION" \
        --query 'Stacks[0].Outputs[?OutputKey==`DataBucketName`].OutputValue' \
        --output text 2>/dev/null)
    
    if [ -n "$bucket_name" ] && [ -d "../../data" ]; then
        info "Uploading to S3 bucket: $bucket_name"
        aws s3 sync ../../data "s3://$bucket_name/sample-data/" --region "$REGION"
        log "Sample data uploaded"
    else
        warn "No sample data directory found or bucket not available"
    fi
}

# Main deploy function
deploy() {
    log "Starting AWS deployment..."
    
    check_dependencies
    
    GITHUB_TOKEN=$(get_github_token)
    ECR_URI=$(setup_ecr)
    
    build_and_push "$ECR_URI"
    deploy_stack "$GITHUB_TOKEN"
    wait_for_service
    upload_sample_data
    
    log "Deployment completed successfully!"
    echo
    info "Stack outputs:"
    get_outputs
    
    echo
    local app_url
    app_url=$(aws cloudformation describe-stacks \
        --stack-name "$STACK_NAME" \
        --region "$REGION" \
        --query 'Stacks[0].Outputs[?OutputKey==`ApplicationURL`].OutputValue' \
        --output text)
    
    log "🚀 Application is available at: $app_url"
}

# Destroy stack and resources
destroy() {
    log "Destroying AWS resources..."
    
    check_dependencies
    
    read -p "Are you sure you want to delete the stack '$STACK_NAME' and all its resources? (y/N): " confirm
    if [[ ! $confirm =~ ^[Yy]$ ]]; then
        log "Operation cancelled"
        exit 0
    fi
    
    # Get bucket name before deletion
    local bucket_name
    bucket_name=$(aws cloudformation describe-stacks \
        --stack-name "$STACK_NAME" \
        --region "$REGION" \
        --query 'Stacks[0].Outputs[?OutputKey==`DataBucketName`].OutputValue' \
        --output text 2>/dev/null || echo "")
    
    # Empty S3 bucket
    if [ -n "$bucket_name" ]; then
        log "Emptying S3 bucket: $bucket_name"
        aws s3 rm "s3://$bucket_name" --recursive --region "$REGION" || true
        
        # Delete all versions if versioning is enabled
        aws s3api list-object-versions --bucket "$bucket_name" --region "$REGION" --query '{Objects: Versions[].{Key: Key, VersionId: VersionId}}' --output json | jq -r '.Objects[] | "--key \"\(.Key)\" --version-id \(.VersionId)"' | xargs -I {} aws s3api delete-object --bucket "$bucket_name" --region "$REGION" {} || true
    fi
    
    # Delete CloudFormation stack
    log "Deleting CloudFormation stack..."
    aws cloudformation delete-stack --stack-name "$STACK_NAME" --region "$REGION"
    
    # Wait for deletion (with timeout)
    log "Waiting for stack deletion (this may take several minutes)..."
    if ! timeout 1200 aws cloudformation wait stack-delete-complete --stack-name "$STACK_NAME" --region "$REGION"; then
        warn "Stack deletion is taking longer than expected. Check AWS console for status."
    else
        log "Stack deleted successfully"
    fi
    
    # Clean up ECR repository
    local repo_name="$STACK_NAME-app"
    if aws ecr describe-repositories --repository-names "$repo_name" --region "$REGION" &> /dev/null; then
        log "Deleting ECR repository..."
        aws ecr delete-repository --repository-name "$repo_name" --force --region "$REGION" || warn "Failed to delete ECR repository"
    fi
    
    log "Destruction complete"
}

# Show detailed status
status() {
    log "Checking deployment status..."
    
    check_dependencies
    
    # Stack status
    echo
    info "CloudFormation Stack Status:"
    if aws cloudformation describe-stacks --stack-name "$STACK_NAME" --region "$REGION" &> /dev/null; then
        aws cloudformation describe-stacks \
            --stack-name "$STACK_NAME" \
            --region "$REGION" \
            --query 'Stacks[0].{StackName:StackName,Status:StackStatus,CreationTime:CreationTime,LastUpdated:LastUpdatedTime}' \
            --output table
    else
        warn "Stack '$STACK_NAME' not found"
        return 1
    fi
    
    # ECS Service status
    echo
    info "ECS Service Status:"
    local cluster_name
    cluster_name=$(aws cloudformation describe-stacks \
        --stack-name "$STACK_NAME" \
        --region "$REGION" \
        --query 'Stacks[0].Outputs[?OutputKey==`ECSClusterName`].OutputValue' \
        --output text 2>/dev/null)
    
    if [ -n "$cluster_name" ]; then
        aws ecs describe-services \
            --cluster "$cluster_name" \
            --services "${STACK_NAME}-service" \
            --region "$REGION" \
            --query 'services[0].{ServiceName:serviceName,Status:status,RunningCount:runningCount,DesiredCount:desiredCount,TaskDefinition:taskDefinition}' \
            --output table
        
        # Task status
        echo
        info "Running Tasks:"
        aws ecs list-tasks \
            --cluster "$cluster_name" \
            --service-name "${STACK_NAME}-service" \
            --region "$REGION" \
            --query 'taskArns' \
            --output table
    fi
    
    # Load balancer health
    echo
    info "Application Health:"
    local app_url
    app_url=$(aws cloudformation describe-stacks \
        --stack-name "$STACK_NAME" \
        --region "$REGION" \
        --query 'Stacks[0].Outputs[?OutputKey==`ApplicationURL`].OutputValue' \
        --output text 2>/dev/null)
    
    if [ -n "$app_url" ]; then
        if curl -f -s --max-time 10 "$app_url" > /dev/null; then
            log "✓ Application is healthy at $app_url"
        else
            warn "✗ Application health check failed at $app_url"
        fi
    fi
    
    echo
    info "Stack Outputs:"
    get_outputs
}

# Stream logs from CloudWatch
logs() {
    log "Streaming logs from CloudWatch..."
    
    check_dependencies
    
    local log_group="/ecs/$STACK_NAME"
    local service_name="${2:-}"
    
    if [ -n "$service_name" ]; then
        # Stream logs for specific service
        aws logs tail "$log_group" --follow --region "$REGION" --filter-pattern "$service_name"
    else
        # Stream all logs
        aws logs tail "$log_group" --follow --region "$REGION"
    fi
}

# Update deployment with new image
update() {
    log "Updating deployment..."
    
    check_dependencies
    
    # Get ECR URI
    local ecr_uri="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${STACK_NAME}-app"
    
    # Build and push new image
    build_and_push "$ecr_uri"
    
    # Force new deployment
    local cluster_name
    cluster_name=$(aws cloudformation describe-stacks \
        --stack-name "$STACK_NAME" \
        --region "$REGION" \
        --query 'Stacks[0].Outputs[?OutputKey==`ECSClusterName`].OutputValue' \
        --output text)
    
    if [ -n "$cluster_name" ]; then
        log "Forcing service update..."
        aws ecs update-service \
            --cluster "$cluster_name" \
            --service "${STACK_NAME}-service" \
            --force-new-deployment \
            --region "$REGION"
        
        wait_for_service
        log "Update complete"
    else
        error "Could not find ECS cluster"
    fi
}

# Scale service up or down
scale() {
    local desired_count="${2:-2}"
    
    log "Scaling service to $desired_count tasks..."
    
    check_dependencies
    
    local cluster_name
    cluster_name=$(aws cloudformation describe-stacks \
        --stack-name "$STACK_NAME" \
        --region "$REGION" \
        --query 'Stacks[0].Outputs[?OutputKey==`ECSClusterName`].OutputValue' \
        --output text)
    
    if [ -n "$cluster_name" ]; then
        aws ecs update-service \
            --cluster "$cluster_name" \
            --service "${STACK_NAME}-service" \
            --desired-count "$desired_count" \
            --region "$REGION"
        
        wait_for_service
        log "Scaling complete"
    else
        error "Could not find ECS cluster"
    fi
}

# Show usage information
usage() {
    cat << EOF
AWS Deployment Script for WhatsApp RAG

Usage: $0 <command> [options]

Commands:
    deploy          Deploy the complete infrastructure
    destroy         Destroy all AWS resources
    status          Show deployment status and health
    logs [service]  Stream CloudWatch logs
    update          Update application with new Docker image
    scale <count>   Scale ECS service to specified task count
    
Environment Variables:
    AWS_REGION      AWS region (default: us-east-1)
    GITHUB_TOKEN    GitHub token for API access
    
Examples:
    $0 deploy                       # Deploy everything
    $0 status                       # Check status
    $0 logs                         # Stream all logs
    $0 logs whatsapp-rag           # Stream app logs only
    $0 scale 4                      # Scale to 4 tasks
    $0 update                       # Update with new image
    $0 destroy                      # Remove all resources
    
Prerequisites:
    - AWS CLI configured with appropriate permissions
    - Docker installed and running
    - GitHub token for API access
    
EOF
}

# Main script execution
main() {
    case "$1" in
        deploy)
            deploy
            ;;
        destroy)
            destroy
            ;;
        status)
            status
            ;;
        logs)
            logs "$@"
            ;;
        update)
            update
            ;;
        scale)
            scale "$@"
            ;;
        *)
            usage
            exit 1
            ;;
    esac
}

# Execute main function with all arguments
main "$@"
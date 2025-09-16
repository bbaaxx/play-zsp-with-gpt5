#!/bin/bash

# WhatsApp RAG Docker Deployment Script
set -e

# Configuration
PROJECT_NAME="whatsapp-rag"
COMPOSE_FILE="docker-compose.yml"
PROD_COMPOSE_FILE="docker-compose.prod.yml"
ENV_FILE=".env"
SSL_DIR="ssl"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[WARNING] $1${NC}"
}

error() {
    echo -e "${RED}[ERROR] $1${NC}"
    exit 1
}

# Check if Docker and Docker Compose are installed
check_dependencies() {
    log "Checking dependencies..."
    
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed. Please install Docker first."
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        error "Docker Compose is not installed. Please install Docker Compose first."
    fi
    
    # Check if Docker daemon is running
    if ! docker info &> /dev/null; then
        error "Docker daemon is not running. Please start Docker."
    fi
    
    log "Dependencies OK"
}

# Create environment file if it doesn't exist
setup_env() {
    log "Setting up environment..."
    
    if [ ! -f "$ENV_FILE" ]; then
        if [ -f "../../.env.example" ]; then
            cp "../../.env.example" "$ENV_FILE"
            log "Created $ENV_FILE from template"
        else
            cat > "$ENV_FILE" << EOF
# WhatsApp RAG Environment Configuration
OPENAI_API_KEY=your_github_token_here
GITHUB_TOKEN=your_github_token_here
MODEL_NAME=gpt-4o-mini
LOG_LEVEL=INFO

# Production settings (uncomment for production)
# GRADIO_USERNAME=admin
# GRADIO_PASSWORD=secure_password_here
# MAX_REQUESTS_PER_MINUTE=60
# ENABLE_METRICS=true

# Monitoring (optional)
# GRAFANA_PASSWORD=admin
EOF
            log "Created basic $ENV_FILE"
        fi
        
        warn "Please edit $ENV_FILE with your GitHub token before proceeding"
        echo "Press any key to continue after editing the file..."
        read -n 1
    fi
    
    # Source environment variables
    set -a
    source "$ENV_FILE"
    set +a
    
    # Validate required variables
    if [ -z "$OPENAI_API_KEY" ] || [ "$OPENAI_API_KEY" = "your_github_token_here" ]; then
        error "OPENAI_API_KEY is not set in $ENV_FILE"
    fi
    
    if [ -z "$GITHUB_TOKEN" ] || [ "$GITHUB_TOKEN" = "your_github_token_here" ]; then
        error "GITHUB_TOKEN is not set in $ENV_FILE"
    fi
    
    log "Environment setup complete"
}

# Generate SSL certificates for development
generate_ssl() {
    log "Setting up SSL certificates..."
    
    mkdir -p "$SSL_DIR"
    
    if [ ! -f "$SSL_DIR/cert.pem" ] || [ ! -f "$SSL_DIR/key.pem" ]; then
        log "Generating self-signed SSL certificates..."
        openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
            -keyout "$SSL_DIR/key.pem" \
            -out "$SSL_DIR/cert.pem" \
            -subj "/C=US/ST=State/L=City/O=Organization/CN=localhost"
        log "SSL certificates generated"
    else
        log "SSL certificates already exist"
    fi
}

# Build and start services
deploy_dev() {
    log "Deploying development environment..."
    
    # Build and start services
    docker-compose -f "$COMPOSE_FILE" build --no-cache
    docker-compose -f "$COMPOSE_FILE" up -d
    
    log "Development deployment complete!"
    log "Application will be available at: http://localhost:7860"
    
    # Show status
    docker-compose -f "$COMPOSE_FILE" ps
}

# Deploy production environment
deploy_prod() {
    log "Deploying production environment..."
    
    # Generate SSL certificates
    generate_ssl
    
    # Build and start services
    docker-compose -f "$PROD_COMPOSE_FILE" build --no-cache
    docker-compose -f "$PROD_COMPOSE_FILE" up -d
    
    log "Production deployment complete!"
    log "Application will be available at:"
    log "  - HTTP: http://localhost"
    log "  - HTTPS: https://localhost"
    log "  - Monitoring: http://localhost:3000 (if monitoring profile enabled)"
    
    # Show status
    docker-compose -f "$PROD_COMPOSE_FILE" ps
}

# Deploy with monitoring
deploy_with_monitoring() {
    log "Deploying with monitoring stack..."
    
    # Generate SSL certificates
    generate_ssl
    
    # Build and start services including monitoring
    docker-compose -f "$PROD_COMPOSE_FILE" --profile monitoring --profile cache build --no-cache
    docker-compose -f "$PROD_COMPOSE_FILE" --profile monitoring --profile cache up -d
    
    log "Deployment with monitoring complete!"
    log "Services available at:"
    log "  - Application: https://localhost"
    log "  - Prometheus: http://localhost:9090"
    log "  - Grafana: http://localhost:3000"
    log "  - Redis: localhost:6379"
}

# Stop services
stop() {
    log "Stopping services..."
    
    if [ -f "$PROD_COMPOSE_FILE" ]; then
        docker-compose -f "$PROD_COMPOSE_FILE" --profile monitoring --profile cache down
    fi
    
    docker-compose -f "$COMPOSE_FILE" down
    
    log "Services stopped"
}

# Destroy all resources
destroy() {
    log "Destroying all resources..."
    
    read -p "This will remove all containers, volumes, and networks. Are you sure? (y/N): " confirm
    if [[ ! $confirm =~ ^[Yy]$ ]]; then
        log "Aborted"
        exit 0
    fi
    
    # Stop and remove everything
    docker-compose -f "$PROD_COMPOSE_FILE" --profile monitoring --profile cache down -v --remove-orphans 2>/dev/null || true
    docker-compose -f "$COMPOSE_FILE" down -v --remove-orphans 2>/dev/null || true
    
    # Remove images
    docker images | grep "$PROJECT_NAME" | awk '{print $3}' | xargs docker rmi -f 2>/dev/null || true
    
    log "All resources destroyed"
}

# Show logs
logs() {
    local service="${2:-}"
    
    if [ -n "$service" ]; then
        if docker-compose -f "$PROD_COMPOSE_FILE" ps | grep -q "$service"; then
            docker-compose -f "$PROD_COMPOSE_FILE" logs -f "$service"
        else
            docker-compose -f "$COMPOSE_FILE" logs -f "$service"
        fi
    else
        # Try production first, then development
        if docker-compose -f "$PROD_COMPOSE_FILE" ps | grep -q "Up"; then
            docker-compose -f "$PROD_COMPOSE_FILE" logs -f
        else
            docker-compose -f "$COMPOSE_FILE" logs -f
        fi
    fi
}

# Show status
status() {
    log "Checking service status..."
    
    echo -e "\n${GREEN}=== Development Services ===${NC}"
    docker-compose -f "$COMPOSE_FILE" ps 2>/dev/null || echo "No development services running"
    
    echo -e "\n${GREEN}=== Production Services ===${NC}"
    docker-compose -f "$PROD_COMPOSE_FILE" ps 2>/dev/null || echo "No production services running"
    
    echo -e "\n${GREEN}=== Docker Resources ===${NC}"
    echo "Images:"
    docker images | grep -E "(whatsapp-rag|nginx|redis|prometheus|grafana)" || echo "No related images found"
    
    echo -e "\nVolumes:"
    docker volume ls | grep "$PROJECT_NAME" || echo "No project volumes found"
    
    echo -e "\nNetworks:"
    docker network ls | grep "$PROJECT_NAME" || echo "No project networks found"
}

# Update services
update() {
    log "Updating services..."
    
    # Pull latest base images
    docker-compose -f "$COMPOSE_FILE" pull
    docker-compose -f "$PROD_COMPOSE_FILE" pull
    
    # Rebuild application image
    docker-compose -f "$COMPOSE_FILE" build --no-cache whatsapp-rag
    docker-compose -f "$PROD_COMPOSE_FILE" build --no-cache whatsapp-rag
    
    # Restart services
    if docker-compose -f "$PROD_COMPOSE_FILE" ps | grep -q "Up"; then
        log "Restarting production services..."
        docker-compose -f "$PROD_COMPOSE_FILE" up -d
    elif docker-compose -f "$COMPOSE_FILE" ps | grep -q "Up"; then
        log "Restarting development services..."
        docker-compose -f "$COMPOSE_FILE" up -d
    fi
    
    log "Update complete"
}

# Health check
health() {
    log "Performing health check..."
    
    local url="http://localhost:7860"
    local max_attempts=30
    local attempt=0
    
    while [ $attempt -lt $max_attempts ]; do
        if curl -f -s "$url" > /dev/null 2>&1; then
            log "✓ Application is healthy at $url"
            return 0
        fi
        
        attempt=$((attempt + 1))
        echo -n "."
        sleep 2
    done
    
    error "✗ Application health check failed after $max_attempts attempts"
}

# Backup data
backup() {
    log "Creating backup..."
    
    local backup_dir="backups/$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$backup_dir"
    
    # Backup volumes
    docker run --rm -v whatsapp-rag_app_logs:/data -v $(pwd)/$backup_dir:/backup alpine \
        sh -c "cd /data && tar czf /backup/logs.tar.gz ."
    
    # Backup configuration
    cp -r . "$backup_dir/config" 2>/dev/null || true
    
    log "Backup created in $backup_dir"
}

# Show usage information
usage() {
    cat << EOF
WhatsApp RAG Docker Deployment Script

Usage: $0 <command> [options]

Commands:
    deploy-dev      Deploy development environment
    deploy-prod     Deploy production environment
    deploy-monitor  Deploy with monitoring stack
    stop           Stop all services
    destroy        Remove all containers, volumes, and networks
    status         Show status of all services
    logs [service] Show logs (optionally for specific service)
    update         Update services to latest version
    health         Check application health
    backup         Create backup of data and configuration
    
Examples:
    $0 deploy-dev                    # Deploy development environment
    $0 deploy-prod                   # Deploy production environment
    $0 deploy-monitor                # Deploy with Prometheus + Grafana
    $0 logs whatsapp-rag            # Show logs for specific service
    $0 status                       # Show status of all services
    
Environment:
    Edit .env file to configure API keys and other settings.
    
Prerequisites:
    - Docker and Docker Compose installed
    - GitHub token for API access
    
EOF
}

# Main script logic
main() {
    case "$1" in
        deploy-dev)
            check_dependencies
            setup_env
            deploy_dev
            health
            ;;
        deploy-prod)
            check_dependencies
            setup_env
            deploy_prod
            health
            ;;
        deploy-monitor)
            check_dependencies
            setup_env
            deploy_with_monitoring
            health
            ;;
        stop)
            stop
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
        health)
            health
            ;;
        backup)
            backup
            ;;
        *)
            usage
            exit 1
            ;;
    esac
}

# Run main function
main "$@"
#!/bin/bash
# WhatsApp RAG (ES) Production Deployment Script
# Usage: ./deploy.sh [environment]

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
ENVIRONMENT="${1:-production}"
SERVICE_NAME="whatsapp-rag"
USER_NAME="whatsapp-rag"
INSTALL_DIR="/opt/whatsapp-rag"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running as root
check_root() {
    if [[ $EUID -ne 0 ]]; then
        log_error "This script must be run as root"
        exit 1
    fi
}

# Create service user
create_service_user() {
    log_info "Creating service user and group..."
    
    if ! getent group "$USER_NAME" > /dev/null 2>&1; then
        groupadd --system "$USER_NAME"
    fi
    
    if ! getent passwd "$USER_NAME" > /dev/null 2>&1; then
        useradd --system --group "$USER_NAME" \
                --home-dir "$INSTALL_DIR" \
                --no-create-home \
                --shell /bin/false \
                "$USER_NAME"
    fi
}

# Install system dependencies
install_system_deps() {
    log_info "Installing system dependencies..."
    
    apt-get update
    apt-get install -y \
        python3.11 \
        python3.11-venv \
        python3.11-dev \
        build-essential \
        curl \
        git \
        nginx \
        certbot \
        python3-certbot-nginx \
        htop \
        tmux \
        ufw
}

# Setup application directory
setup_app_directory() {
    log_info "Setting up application directory..."
    
    # Create directory structure
    mkdir -p "$INSTALL_DIR"/{data,logs,config}
    
    # Copy application files
    cp -r "$PROJECT_ROOT"/* "$INSTALL_DIR"/
    
    # Set permissions
    chown -R "$USER_NAME:$USER_NAME" "$INSTALL_DIR"
    chmod -R 755 "$INSTALL_DIR"
    chmod 600 "$INSTALL_DIR/.env" 2>/dev/null || true
}

# Setup Python environment
setup_python_env() {
    log_info "Setting up Python virtual environment..."
    
    cd "$INSTALL_DIR"
    
    # Create virtual environment as service user
    sudo -u "$USER_NAME" python3.11 -m venv .venv
    
    # Install dependencies
    sudo -u "$USER_NAME" .venv/bin/pip install --upgrade pip
    sudo -u "$USER_NAME" .venv/bin/pip install -r requirements.txt
}

# Configure environment
configure_environment() {
    log_info "Configuring environment..."
    
    # Copy environment template if .env doesn't exist
    if [[ ! -f "$INSTALL_DIR/.env" ]]; then
        if [[ -f "$INSTALL_DIR/config/${ENVIRONMENT}.env.template" ]]; then
            cp "$INSTALL_DIR/config/${ENVIRONMENT}.env.template" "$INSTALL_DIR/.env"
            log_warn "Environment file created from template. Please edit $INSTALL_DIR/.env"
        else
            cp "$INSTALL_DIR/.env.example" "$INSTALL_DIR/.env"
            log_warn "Environment file created from example. Please edit $INSTALL_DIR/.env"
        fi
        
        # Set secure permissions
        chown "$USER_NAME:$USER_NAME" "$INSTALL_DIR/.env"
        chmod 600 "$INSTALL_DIR/.env"
        
        echo
        log_warn "IMPORTANT: Edit $INSTALL_DIR/.env with your configuration before starting the service"
        echo "Required: GITHUB_TOKEN"
        echo
    fi
}

# Install systemd service
install_systemd_service() {
    log_info "Installing systemd service..."
    
    # Copy service file
    cp "$INSTALL_DIR/config/systemd/whatsapp-rag.service" "/etc/systemd/system/"
    
    # Reload systemd
    systemctl daemon-reload
    
    # Enable service
    systemctl enable "$SERVICE_NAME"
    
    log_info "Service installed and enabled"
}

# Configure firewall
configure_firewall() {
    log_info "Configuring firewall..."
    
    # Enable UFW if not already active
    ufw --force enable
    
    # Allow SSH, HTTP, HTTPS
    ufw allow ssh
    ufw allow 80/tcp
    ufw allow 443/tcp
    
    # Block direct access to Gradio port (only allow via reverse proxy)
    ufw deny 7860/tcp
    
    log_info "Firewall configured"
}

# Setup Nginx
setup_nginx() {
    log_info "Setting up Nginx reverse proxy..."
    
    # Copy nginx config
    cp "$INSTALL_DIR/config/nginx.conf" "/etc/nginx/sites-available/$SERVICE_NAME"
    
    # Create symlink
    ln -sf "/etc/nginx/sites-available/$SERVICE_NAME" "/etc/nginx/sites-enabled/"
    
    # Remove default site
    rm -f /etc/nginx/sites-enabled/default
    
    # Test nginx configuration
    nginx -t
    
    # Reload nginx
    systemctl reload nginx
    
    log_warn "Please edit /etc/nginx/sites-available/$SERVICE_NAME to set your domain name"
    log_warn "Then run: certbot --nginx -d your-domain.com"
}

# Health check
health_check() {
    log_info "Performing health check..."
    
    # Start the service
    systemctl start "$SERVICE_NAME"
    
    # Wait for service to start
    sleep 10
    
    # Check service status
    if systemctl is-active --quiet "$SERVICE_NAME"; then
        log_info "Service is running"
    else
        log_error "Service failed to start"
        systemctl status "$SERVICE_NAME"
        return 1
    fi
    
    # Check if port is listening
    if ss -tuln | grep -q ":7860"; then
        log_info "Application is listening on port 7860"
    else
        log_error "Application is not listening on expected port"
        return 1
    fi
    
    # Test HTTP endpoint
    if curl -f -s http://localhost:7860 > /dev/null; then
        log_info "HTTP endpoint is responding"
    else
        log_error "HTTP endpoint is not responding"
        return 1
    fi
}

# Display post-deployment information
post_deployment_info() {
    echo
    log_info "=== Deployment Summary ==="
    echo "Service: $SERVICE_NAME"
    echo "Install directory: $INSTALL_DIR"
    echo "User: $USER_NAME"
    echo "Environment: $ENVIRONMENT"
    echo
    log_info "=== Next Steps ==="
    echo "1. Edit $INSTALL_DIR/.env with your configuration"
    echo "2. Edit /etc/nginx/sites-available/$SERVICE_NAME with your domain"
    echo "3. Set up SSL: certbot --nginx -d your-domain.com"
    echo "4. Restart services: systemctl restart $SERVICE_NAME nginx"
    echo
    log_info "=== Useful Commands ==="
    echo "Status:  systemctl status $SERVICE_NAME"
    echo "Logs:    journalctl -u $SERVICE_NAME -f"
    echo "Restart: systemctl restart $SERVICE_NAME"
    echo "Stop:    systemctl stop $SERVICE_NAME"
    echo
}

# Cleanup on error
cleanup_on_error() {
    log_error "Deployment failed. Cleaning up..."
    
    # Stop service if it was started
    systemctl stop "$SERVICE_NAME" 2>/dev/null || true
    systemctl disable "$SERVICE_NAME" 2>/dev/null || true
    
    # Remove systemd service file
    rm -f "/etc/systemd/system/$SERVICE_NAME.service"
    
    # Remove nginx config
    rm -f "/etc/nginx/sites-enabled/$SERVICE_NAME"
    rm -f "/etc/nginx/sites-available/$SERVICE_NAME"
    
    systemctl daemon-reload
    
    exit 1
}

# Main deployment function
deploy() {
    log_info "Starting deployment for environment: $ENVIRONMENT"
    
    # Set trap for cleanup on error
    trap cleanup_on_error ERR
    
    check_root
    create_service_user
    install_system_deps
    setup_app_directory
    setup_python_env
    configure_environment
    install_systemd_service
    configure_firewall
    setup_nginx
    health_check
    post_deployment_info
    
    log_info "Deployment completed successfully!"
}

# Show usage
show_usage() {
    echo "Usage: $0 [environment]"
    echo
    echo "Environments:"
    echo "  production  - Production deployment (default)"
    echo "  development - Development deployment"
    echo
    echo "Example:"
    echo "  $0 production"
}

# Parse command line arguments
case "${1:-}" in
    -h|--help)
        show_usage
        exit 0
        ;;
    production|development)
        deploy
        ;;
    "")
        deploy
        ;;
    *)
        log_error "Unknown environment: $1"
        show_usage
        exit 1
        ;;
esac
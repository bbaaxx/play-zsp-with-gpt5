#!/bin/bash
# Health check script for WhatsApp RAG (ES)
# Can be used for monitoring systems like Nagios, Zabbix, etc.

set -euo pipefail

# Configuration
SERVICE_NAME="whatsapp-rag"
HOST="${1:-localhost}"
PORT="${2:-7860}"
TIMEOUT="${3:-10}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Exit codes
EXIT_OK=0
EXIT_WARNING=1
EXIT_CRITICAL=2
EXIT_UNKNOWN=3

# Functions
log_ok() {
    echo -e "${GREEN}[OK]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_critical() {
    echo -e "${RED}[CRITICAL]${NC} $1"
}

# Check if service is running
check_service() {
    if systemctl is-active --quiet "$SERVICE_NAME"; then
        log_ok "Service $SERVICE_NAME is running"
        return 0
    else
        log_critical "Service $SERVICE_NAME is not running"
        return $EXIT_CRITICAL
    fi
}

# Check if port is listening
check_port() {
    if timeout "$TIMEOUT" bash -c "echo > /dev/tcp/$HOST/$PORT" 2>/dev/null; then
        log_ok "Port $PORT is listening"
        return 0
    else
        log_critical "Port $PORT is not accessible"
        return $EXIT_CRITICAL
    fi
}

# Check HTTP endpoint
check_http() {
    local response
    local http_code
    
    response=$(curl -s -w "HTTPSTATUS:%{http_code}" \
                   --max-time "$TIMEOUT" \
                   "http://$HOST:$PORT" || true)
    
    http_code=$(echo "$response" | tr -d '\n' | sed -e 's/.*HTTPSTATUS://')
    
    if [[ "$http_code" -eq 200 ]]; then
        log_ok "HTTP endpoint responds with 200"
        return 0
    elif [[ "$http_code" -ge 400 && "$http_code" -lt 500 ]]; then
        log_warning "HTTP endpoint responds with $http_code (client error)"
        return $EXIT_WARNING
    else
        log_critical "HTTP endpoint responds with $http_code or no response"
        return $EXIT_CRITICAL
    fi
}

# Check memory usage
check_memory() {
    local memory_usage
    local pid
    
    pid=$(pgrep -f "python.*app.py" || echo "")
    
    if [[ -z "$pid" ]]; then
        log_warning "Cannot find Python process"
        return $EXIT_WARNING
    fi
    
    # Get memory usage in MB
    memory_usage=$(ps -o rss= -p "$pid" 2>/dev/null | awk '{print int($1/1024)}' || echo "0")
    
    if [[ "$memory_usage" -gt 2048 ]]; then
        log_warning "High memory usage: ${memory_usage}MB"
        return $EXIT_WARNING
    elif [[ "$memory_usage" -gt 4096 ]]; then
        log_critical "Very high memory usage: ${memory_usage}MB"
        return $EXIT_CRITICAL
    else
        log_ok "Memory usage normal: ${memory_usage}MB"
        return 0
    fi
}

# Check disk space
check_disk() {
    local disk_usage
    local install_dir="/opt/whatsapp-rag"
    
    if [[ ! -d "$install_dir" ]]; then
        install_dir="/tmp"
    fi
    
    disk_usage=$(df "$install_dir" | tail -1 | awk '{print $5}' | sed 's/%//')
    
    if [[ "$disk_usage" -gt 90 ]]; then
        log_critical "Disk usage critical: ${disk_usage}%"
        return $EXIT_CRITICAL
    elif [[ "$disk_usage" -gt 80 ]]; then
        log_warning "Disk usage high: ${disk_usage}%"
        return $EXIT_WARNING
    else
        log_ok "Disk usage normal: ${disk_usage}%"
        return 0
    fi
}

# Check log files for errors
check_logs() {
    local error_count
    local log_file="/var/log/syslog"
    
    # Count recent errors (last 5 minutes)
    error_count=$(journalctl -u "$SERVICE_NAME" --since "5 minutes ago" --priority=err --no-pager -q | wc -l 2>/dev/null || echo "0")
    
    if [[ "$error_count" -gt 10 ]]; then
        log_critical "$error_count errors in the last 5 minutes"
        return $EXIT_CRITICAL
    elif [[ "$error_count" -gt 5 ]]; then
        log_warning "$error_count errors in the last 5 minutes"
        return $EXIT_WARNING
    else
        log_ok "No significant errors in logs"
        return 0
    fi
}

# Main health check
main() {
    local overall_status=$EXIT_OK
    local check_status
    
    echo "=== WhatsApp RAG Health Check ==="
    echo "Host: $HOST"
    echo "Port: $PORT"
    echo "Service: $SERVICE_NAME"
    echo "Timeout: ${TIMEOUT}s"
    echo

    # Run all checks
    checks=(
        "Service Status:check_service"
        "Port Accessibility:check_port"
        "HTTP Response:check_http"
        "Memory Usage:check_memory"
        "Disk Usage:check_disk"
        "Log Errors:check_logs"
    )

    for check in "${checks[@]}"; do
        check_name="${check%%:*}"
        check_func="${check##*:}"
        
        echo "Checking $check_name..."
        
        if $check_func; then
            check_status=$?
        else
            check_status=$?
        fi
        
        # Track worst status
        if [[ $check_status -gt $overall_status ]]; then
            overall_status=$check_status
        fi
        
        echo
    done

    # Summary
    case $overall_status in
        $EXIT_OK)
            echo -e "${GREEN}Overall Status: HEALTHY${NC}"
            ;;
        $EXIT_WARNING)
            echo -e "${YELLOW}Overall Status: WARNING${NC}"
            ;;
        $EXIT_CRITICAL)
            echo -e "${RED}Overall Status: CRITICAL${NC}"
            ;;
        $EXIT_UNKNOWN)
            echo -e "${YELLOW}Overall Status: UNKNOWN${NC}"
            ;;
    esac

    exit $overall_status
}

# Show usage
show_usage() {
    echo "Usage: $0 [host] [port] [timeout]"
    echo
    echo "Parameters:"
    echo "  host     - Host to check (default: localhost)"
    echo "  port     - Port to check (default: 7860)"
    echo "  timeout  - Timeout in seconds (default: 10)"
    echo
    echo "Exit codes:"
    echo "  0 - OK (healthy)"
    echo "  1 - WARNING (minor issues)"
    echo "  2 - CRITICAL (major issues)"
    echo "  3 - UNKNOWN (check failed)"
    echo
    echo "Examples:"
    echo "  $0                    # Check localhost:7860"
    echo "  $0 example.com 80     # Check example.com:80"
    echo "  $0 localhost 7860 30  # Check with 30s timeout"
}

# Parse arguments
case "${1:-}" in
    -h|--help)
        show_usage
        exit 0
        ;;
    *)
        main
        ;;
esac
# WhatsApp RAG (ES) - Production Deployment Guide

## Table of Contents
1. [System Overview](#system-overview)
2. [Prerequisites](#prerequisites)
3. [Environment Configuration](#environment-configuration)
4. [Installation](#installation)
5. [Production Configuration](#production-configuration)
6. [Performance Tuning](#performance-tuning)
7. [Security Best Practices](#security-best-practices)
8. [Monitoring and Logging](#monitoring-and-logging)
9. [Deployment Options](#deployment-options)
10. [Troubleshooting](#troubleshooting)
11. [Maintenance](#maintenance)

## System Overview

The WhatsApp RAG system is a Python-based application that provides semantic search and Q&A capabilities over WhatsApp chat exports. It consists of:

- **Web Interface**: Gradio-based web UI
- **RAG Pipeline**: Custom implementation with chunking, embeddings, and vector storage
- **Vector Store**: FAISS-based similarity search with numpy fallback
- **LLM Integration**: OpenAI API via GitHub Models
- **Embedding Providers**: Remote (GitHub Models) with local fallback (sentence-transformers)

## Prerequisites

### System Requirements

**Minimum Requirements:**
- CPU: 2 cores
- RAM: 4GB
- Storage: 10GB free space
- Python: 3.11+

**Recommended for Production:**
- CPU: 4+ cores
- RAM: 8GB+
- Storage: 50GB+ SSD
- Network: Stable internet connection for API calls

### Dependencies
- Python 3.11 or higher
- pip package manager
- Git (for deployment)
- SSL certificates (for HTTPS deployment)

## Environment Configuration

### Required Environment Variables

Create a `.env` file in the project root with the following variables:

```bash
# GitHub Models API Configuration (REQUIRED)
GITHUB_TOKEN=ghp_your_github_personal_access_token_here
GH_MODELS_BASE_URL=https://models.github.ai/inference

# Model Configuration
CHAT_MODEL=openai/gpt-4o
EMBEDDING_MODEL=openai/text-embedding-3-small

# Embedding Fallback Configuration
USE_LOCAL_EMBEDDINGS=0
LOCAL_EMBEDDING_MODEL=intfloat/multilingual-e5-small

# Performance Tuning
EMBED_BATCH_SIZE=64
EMBED_MAX_CHARS_PER_REQUEST=60000
EMBED_MAX_CHARS_PER_ITEM=4000

# Server Configuration
HOST=0.0.0.0
PORT=7860
GRADIO_SHARE=0
GRADIO_ANALYTICS_ENABLED=0

# Optional: Logging
LOG_LEVEL=INFO
```

### Environment Variables Reference

#### Core API Configuration

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GITHUB_TOKEN` | Yes | - | GitHub Personal Access Token with access to GitHub Models |
| `GH_MODELS_BASE_URL` | No | `https://models.github.ai/inference` | Base URL for GitHub Models API |
| `CHAT_MODEL` | No | `openai/gpt-4o` | LLM model for chat completions |
| `EMBEDDING_MODEL` | No | `openai/text-embedding-3-small` | Remote embedding model |

#### Embedding Configuration

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `USE_LOCAL_EMBEDDINGS` | No | `0` | Force use of local embeddings (1=yes, 0=no) |
| `LOCAL_EMBEDDING_MODEL` | No | `intfloat/multilingual-e5-small` | Local fallback embedding model |
| `EMBED_BATCH_SIZE` | No | `64` | Maximum texts per embedding API request |
| `EMBED_MAX_CHARS_PER_REQUEST` | No | `60000` | Maximum characters per API request |
| `EMBED_MAX_CHARS_PER_ITEM` | No | `4000` | Maximum characters per individual text |

#### Server Configuration

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `HOST` | No | `127.0.0.1` | Server bind address (use `0.0.0.0` for public access) |
| `PORT` | No | `7860` | Server port |
| `GRADIO_SHARE` | No | `0` | Enable Gradio sharing (1=yes, 0=no) |
| `GRADIO_ANALYTICS_ENABLED` | No | `0` | Enable Gradio analytics |
| `LOG_LEVEL` | No | `INFO` | Logging level (DEBUG, INFO, WARNING, ERROR) |

## Installation

### Step 1: Clone Repository

```bash
git clone <repository-url>
cd whatsapp-rag-es
```

### Step 2: Create Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate  # Linux/macOS
# or
.venv\Scripts\activate     # Windows
```

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Configure Environment

```bash
cp .env.example .env
# Edit .env with your configuration
nano .env
```

### Step 5: Test Installation

```bash
python app.py
```

Access the application at `http://localhost:7860`

## Production Configuration

### GitHub Models API Setup

1. **Create GitHub Personal Access Token:**
   - Go to GitHub Settings → Developer settings → Personal access tokens
   - Click "Generate new token (classic)"
   - Select appropriate scopes (typically `repo` and `read:user`)
   - Copy the token to `GITHUB_TOKEN` in `.env`

2. **Verify API Access:**
   ```bash
   curl -H "Authorization: Bearer $GITHUB_TOKEN" \
        -H "Content-Type: application/json" \
        -d '{"model": "openai/gpt-4o", "messages": [{"role": "user", "content": "test"}]}' \
        https://models.github.ai/inference/chat/completions
   ```

### Embedding Provider Configuration

#### Remote Embeddings (Recommended)

```bash
# .env
USE_LOCAL_EMBEDDINGS=0
EMBEDDING_MODEL=openai/text-embedding-3-small
EMBED_BATCH_SIZE=64
EMBED_MAX_CHARS_PER_REQUEST=60000
```

**Benefits:**
- Higher quality embeddings
- No local model storage
- Faster startup
- Consistent across deployments

#### Local Embeddings (Fallback)

```bash
# .env
USE_LOCAL_EMBEDDINGS=1
LOCAL_EMBEDDING_MODEL=intfloat/multilingual-e5-small
```

**Benefits:**
- No API dependencies
- Privacy (embeddings stay local)
- No usage costs
- Works offline

#### Automatic Failover

The system automatically falls back to local embeddings if:
- `GITHUB_TOKEN` is missing
- Remote API is unreachable
- API quota is exceeded
- Network errors occur

## Performance Tuning

### Embedding Performance

```bash
# High-throughput configuration
EMBED_BATCH_SIZE=128
EMBED_MAX_CHARS_PER_REQUEST=100000
EMBED_MAX_CHARS_PER_ITEM=8000

# Conservative configuration (for rate limits)
EMBED_BATCH_SIZE=32
EMBED_MAX_CHARS_PER_REQUEST=30000
EMBED_MAX_CHARS_PER_ITEM=2000
```

### Vector Store Optimization

For large datasets (>10,000 messages):

```python
# In production, consider these chunking parameters:
window_size = 20      # Smaller windows for precision
window_overlap = 5    # Less overlap for performance
```

### Memory Management

Monitor memory usage with:

```bash
# Check memory usage
ps aux | grep python
htop

# For memory-constrained environments
export PYTHONMALLOC=malloc
export MALLOC_MMAP_THRESHOLD_=1024
```

### LLM Performance Tuning

```bash
# Fast responses
CHAT_MODEL=openai/gpt-4o-mini

# High quality (slower)
CHAT_MODEL=openai/gpt-4o

# Balanced
CHAT_MODEL=openai/gpt-4o
```

## Security Best Practices

### API Key Security

1. **Never commit API keys to version control:**
   ```bash
   # Ensure .env is in .gitignore
   echo ".env" >> .gitignore
   ```

2. **Use environment variables in production:**
   ```bash
   # Set via system environment, not .env file
   export GITHUB_TOKEN="ghp_your_token_here"
   ```

3. **Rotate API keys regularly:**
   - GitHub tokens should be rotated every 90 days
   - Use GitHub's token expiration features

### Network Security

1. **Use HTTPS in production:**
   ```bash
   # Behind reverse proxy (recommended)
   HOST=127.0.0.1
   PORT=7860
   
   # Direct HTTPS (configure SSL certificates)
   # Gradio doesn't support HTTPS directly - use nginx/Apache
   ```

2. **Firewall configuration:**
   ```bash
   # Only allow necessary ports
   ufw allow ssh
   ufw allow 80/tcp
   ufw allow 443/tcp
   ufw deny 7860/tcp  # Block direct access to Gradio
   ```

### Data Security

1. **Sanitize uploaded files:**
   - The system already filters system messages
   - Consider additional input validation

2. **Secure file storage:**
   ```bash
   # Set restrictive permissions
   chmod 600 .env
   chmod 700 data/
   ```

3. **Privacy considerations:**
   - WhatsApp exports contain personal data
   - Implement data retention policies
   - Consider GDPR compliance

## Monitoring and Logging

### Application Logging

Configure structured logging:

```python
# Enhanced logging configuration
import logging
import sys
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('whatsapp_rag.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
```

### Health Monitoring

Create a health check endpoint:

```bash
# Check if service is responsive
curl -f http://localhost:7860 || exit 1
```

### Resource Monitoring

Monitor key metrics:

```bash
# CPU and memory
top -p $(pgrep -f "python app.py")

# Disk usage
df -h
du -sh data/

# Network connections
netstat -an | grep :7860
```

### Log Analysis

Common log patterns to monitor:

```bash
# API failures
grep "Error al llamar al modelo" whatsapp_rag.log

# Embedding fallbacks
grep "fallback local" whatsapp_rag.log

# Memory issues
dmesg | grep -i "killed process"
```

## Deployment Options

### 1. Systemd Service (Linux)

Create `/etc/systemd/system/whatsapp-rag.service`:

```ini
[Unit]
Description=WhatsApp RAG Service
After=network.target

[Service]
Type=simple
User=whatsapp-rag
WorkingDirectory=/opt/whatsapp-rag
Environment=PATH=/opt/whatsapp-rag/.venv/bin
ExecStart=/opt/whatsapp-rag/.venv/bin/python app.py
EnvironmentFile=/opt/whatsapp-rag/.env
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
# Enable and start service
sudo systemctl enable whatsapp-rag
sudo systemctl start whatsapp-rag
sudo systemctl status whatsapp-rag
```

### 2. Docker Deployment

Create `Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860

CMD ["python", "app.py"]
```

Create `docker-compose.yml`:

```yaml
version: '3.8'
services:
  whatsapp-rag:
    build: .
    ports:
      - "7860:7860"
    environment:
      - GITHUB_TOKEN=${GITHUB_TOKEN}
      - HOST=0.0.0.0
    env_file:
      - .env
    volumes:
      - ./data:/app/data
    restart: unless-stopped
```

Deploy:

```bash
docker-compose up -d
```

### 3. Reverse Proxy Setup (Nginx)

Create `/etc/nginx/sites-available/whatsapp-rag`:

```nginx
server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        proxy_pass http://127.0.0.1:7860;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # WebSocket support for Gradio
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

Enable and configure SSL:

```bash
sudo ln -s /etc/nginx/sites-available/whatsapp-rag /etc/nginx/sites-enabled/
sudo certbot --nginx -d your-domain.com
sudo systemctl reload nginx
```

### 4. Cloud Deployment

#### AWS EC2 Example

```bash
# Launch EC2 instance (Ubuntu 22.04)
# Install dependencies
sudo apt update
sudo apt install python3.11 python3.11-venv nginx certbot

# Deploy application
git clone <repo>
cd whatsapp-rag-es
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with production values

# Set up systemd service (as above)
# Configure nginx reverse proxy
# Set up SSL with certbot
```

## Troubleshooting

### Common Issues

#### 1. Import Errors

**Problem**: `ModuleNotFoundError` or missing dependencies

**Solution**:
```bash
# Recreate virtual environment
rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

#### 2. API Authentication Failures

**Problem**: "GITHUB_TOKEN ausente" or API errors

**Solution**:
```bash
# Verify token is set
echo $GITHUB_TOKEN

# Test API access
curl -H "Authorization: Bearer $GITHUB_TOKEN" \
     https://models.github.ai/inference/models

# Check token permissions in GitHub settings
```

#### 3. Memory Issues

**Problem**: OOM kills or high memory usage

**Solution**:
```bash
# Reduce batch sizes
EMBED_BATCH_SIZE=32
EMBED_MAX_CHARS_PER_REQUEST=30000

# Monitor memory
watch -n 1 'ps aux | grep python'

# Consider swap space
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

#### 4. Embedding Failures

**Problem**: Embedding API errors or timeouts

**Solution**:
```bash
# Enable local fallback
USE_LOCAL_EMBEDDINGS=1

# Or reduce request sizes
EMBED_BATCH_SIZE=16
EMBED_MAX_CHARS_PER_REQUEST=15000
```

#### 5. Port Binding Issues

**Problem**: "Address already in use" error

**Solution**:
```bash
# Find process using port
sudo lsof -i :7860
sudo kill -9 <PID>

# Or use different port
PORT=8080
```

### Performance Debugging

#### Enable Debug Logging

```bash
LOG_LEVEL=DEBUG
python app.py
```

#### Profile Memory Usage

```python
# Add to app.py for debugging
import tracemalloc
import gc

tracemalloc.start()
# ... after operations
current, peak = tracemalloc.get_traced_memory()
print(f"Current: {current / 1024 / 1024:.1f} MB")
print(f"Peak: {peak / 1024 / 1024:.1f} MB")
```

#### Monitor API Latency

```bash
# Time API calls
time curl -X POST http://localhost:7860/api/predict \
  -H "Content-Type: application/json" \
  -d '{"data": ["test message", 5, "openai/gpt-4o", true, 0.5, 25, "", "", ""]}'
```

### Backup and Recovery

#### Data Backup

```bash
# Backup vector indices
tar -czf backup-$(date +%Y%m%d).tar.gz data/ .env

# Automated backup script
#!/bin/bash
BACKUP_DIR="/var/backups/whatsapp-rag"
DATE=$(date +%Y%m%d-%H%M)
mkdir -p $BACKUP_DIR
tar -czf "$BACKUP_DIR/backup-$DATE.tar.gz" data/ .env
find $BACKUP_DIR -name "backup-*.tar.gz" -mtime +7 -delete
```

#### Service Recovery

```bash
# Restart service
sudo systemctl restart whatsapp-rag

# Check logs
sudo journalctl -u whatsapp-rag -f

# Rollback deployment
git checkout previous-stable-commit
sudo systemctl restart whatsapp-rag
```

## Maintenance

### Regular Tasks

1. **Update Dependencies (Monthly)**:
   ```bash
   pip list --outdated
   pip install --upgrade gradio openai sentence-transformers
   pip freeze > requirements.txt
   ```

2. **Monitor Disk Usage**:
   ```bash
   # Clean old logs
   find . -name "*.log" -mtime +30 -delete
   
   # Clean temporary files
   find /tmp -name "*whatsapp*" -mtime +1 -delete
   ```

3. **Security Updates**:
   ```bash
   # System updates
   sudo apt update && sudo apt upgrade
   
   # Python security updates
   pip audit
   ```

4. **Performance Monitoring**:
   ```bash
   # Weekly performance report
   echo "=== Performance Report $(date) ===" >> performance.log
   top -bn1 | grep python >> performance.log
   df -h >> performance.log
   ```

### Scaling Considerations

For high-traffic deployments:

1. **Load Balancing**: Deploy multiple instances behind nginx
2. **Caching**: Implement Redis for query caching
3. **Database**: Move to persistent vector database (Qdrant, Weaviate)
4. **CDN**: Use CloudFlare for static assets
5. **Monitoring**: Implement Prometheus/Grafana

This completes the comprehensive deployment guide. The system is now production-ready with proper monitoring, security, and maintenance procedures.
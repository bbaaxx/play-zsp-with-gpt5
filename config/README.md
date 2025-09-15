# WhatsApp RAG (ES) - Configuration Files

This directory contains configuration templates and deployment files for the WhatsApp RAG system.

## Directory Structure

```
config/
├── README.md                    # This file
├── production.env.template      # Production environment variables
├── development.env.template     # Development environment variables
├── docker-compose.yml          # Docker Compose configuration
├── Dockerfile                  # Multi-stage Docker build
├── nginx.conf                  # Nginx reverse proxy configuration
├── systemd/
│   └── whatsapp-rag.service   # Systemd service file
├── monitoring/
│   └── prometheus.yml         # Prometheus monitoring configuration
└── scripts/
    ├── deploy.sh              # Production deployment script
    └── health-check.sh        # Health monitoring script
```

## Quick Start

### 1. Environment Configuration

For **production**:
```bash
cp config/production.env.template .env
# Edit .env with your configuration
nano .env
```

For **development**:
```bash
cp config/development.env.template .env
# Edit .env with your configuration
nano .env
```

### 2. Docker Deployment

```bash
# Copy environment template
cp config/production.env.template .env

# Edit configuration
nano .env

# Deploy with Docker Compose
docker-compose -f config/docker-compose.yml up -d
```

### 3. Manual Deployment (Ubuntu/Debian)

```bash
# Run deployment script as root
sudo config/scripts/deploy.sh production
```

## Configuration Files Reference

### Environment Templates

#### production.env.template
- Optimized for production deployment
- Remote embeddings enabled by default
- Conservative performance settings
- Security-focused defaults

#### development.env.template
- Optimized for local development
- Local embeddings enabled by default
- Debug logging enabled
- Development-friendly settings

### Docker Configuration

#### docker-compose.yml
- Production-ready Docker Compose setup
- Includes optional nginx reverse proxy
- Monitoring with Prometheus and Grafana
- Resource limits and health checks
- Volume mounting for persistence

#### Dockerfile
- Multi-stage build for optimized image size
- Security hardening with non-root user
- Health checks and proper signal handling
- Production and secure variants

### System Configuration

#### nginx.conf
- SSL/TLS termination
- Rate limiting and security headers
- WebSocket support for Gradio
- Gzip compression and caching
- Security best practices

#### systemd/whatsapp-rag.service
- Systemd service configuration
- Automatic restart and recovery
- Security restrictions and resource limits
- Proper logging and process management

### Monitoring

#### monitoring/prometheus.yml
- Prometheus scraping configuration
- Application and system metrics
- Alert manager integration
- Multi-target monitoring setup

### Scripts

#### scripts/deploy.sh
- Automated production deployment
- System dependency installation
- User and directory setup
- Service configuration and activation

#### scripts/health-check.sh
- Comprehensive health monitoring
- Service, port, and HTTP checks
- Resource usage monitoring
- Log error detection

## Environment Variables Reference

### Required Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `GITHUB_TOKEN` | GitHub Personal Access Token | `ghp_xxxxxxxxxxxx` |

### Core Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `GH_MODELS_BASE_URL` | `https://models.github.ai/inference` | GitHub Models API endpoint |
| `CHAT_MODEL` | `openai/gpt-4o` | LLM model for chat |
| `EMBEDDING_MODEL` | `openai/text-embedding-3-small` | Remote embedding model |

### Performance Tuning

| Variable | Default | Description |
|----------|---------|-------------|
| `EMBED_BATCH_SIZE` | `64` | Texts per embedding request |
| `EMBED_MAX_CHARS_PER_REQUEST` | `60000` | Max characters per request |
| `EMBED_MAX_CHARS_PER_ITEM` | `4000` | Max characters per text |

### Server Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `HOST` | `127.0.0.1` | Server bind address |
| `PORT` | `7860` | Server port |
| `LOG_LEVEL` | `INFO` | Logging verbosity |

## Deployment Scenarios

### 1. Single Server Deployment

```bash
# Manual deployment
sudo config/scripts/deploy.sh production

# Or with Docker
docker-compose -f config/docker-compose.yml up -d
```

### 2. Load Balanced Deployment

```bash
# Deploy multiple instances
docker-compose -f config/docker-compose.yml up -d --scale whatsapp-rag=3

# Use nginx upstream configuration for load balancing
```

### 3. Development Setup

```bash
cp config/development.env.template .env
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

### 4. Monitoring Setup

```bash
# Enable monitoring profile
docker-compose -f config/docker-compose.yml --profile monitoring up -d

# Access Grafana at http://localhost:3000 (admin/admin)
# Access Prometheus at http://localhost:9090
```

## Security Considerations

1. **API Keys**: Never commit `.env` files to version control
2. **Firewall**: Block direct access to application ports
3. **SSL/TLS**: Use HTTPS in production with valid certificates
4. **Updates**: Regularly update dependencies and system packages
5. **Monitoring**: Set up alerts for security events and errors

## Troubleshooting

### Common Issues

1. **Service won't start**:
   ```bash
   journalctl -u whatsapp-rag -f
   ```

2. **Port conflicts**:
   ```bash
   sudo lsof -i :7860
   ```

3. **Permission issues**:
   ```bash
   sudo chown -R whatsapp-rag:whatsapp-rag /opt/whatsapp-rag
   ```

4. **SSL certificate issues**:
   ```bash
   sudo certbot --nginx -d your-domain.com
   ```

### Health Monitoring

Run the health check script:
```bash
config/scripts/health-check.sh
```

Check specific components:
```bash
# Service status
systemctl status whatsapp-rag

# Nginx status
systemctl status nginx

# Application logs
journalctl -u whatsapp-rag -f

# Resource usage
htop
```

## Performance Optimization

### For High Traffic

1. **Scale horizontally**:
   ```bash
   docker-compose up -d --scale whatsapp-rag=5
   ```

2. **Optimize embedding settings**:
   ```env
   EMBED_BATCH_SIZE=128
   EMBED_MAX_CHARS_PER_REQUEST=100000
   ```

3. **Use caching**:
   - Implement Redis for query caching
   - Use CDN for static assets

### For Resource-Constrained Environments

1. **Use local embeddings**:
   ```env
   USE_LOCAL_EMBEDDINGS=1
   ```

2. **Reduce batch sizes**:
   ```env
   EMBED_BATCH_SIZE=16
   EMBED_MAX_CHARS_PER_REQUEST=15000
   ```

3. **Optimize memory usage**:
   ```env
   PYTHONHASHSEED=random
   MALLOC_MMAP_THRESHOLD_=1024
   ```

## Support

For deployment issues:
1. Check the logs with `journalctl -u whatsapp-rag -f`
2. Run the health check script
3. Verify environment configuration
4. Check firewall and network settings
5. Review the main deployment guide in `DEPLOYMENT_GUIDE.md`
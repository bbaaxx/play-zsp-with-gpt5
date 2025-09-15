# Multi-stage Docker build for WhatsApp RAG (ES)
# Optimized for production deployment with minimal image size

# Build stage for dependencies
FROM python:3.11-slim as builder

# Set build arguments
ARG DEBIAN_FRONTEND=noninteractive

# Install system dependencies needed for building
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies in virtual environment
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.11-slim as production

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    GRADIO_SERVER_NAME="0.0.0.0"

# Create non-root user for security
RUN groupadd -r whatsapp && useradd -r -g whatsapp whatsapp

# Install minimal runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# Set working directory
WORKDIR /app

# Copy application code
COPY --chown=whatsapp:whatsapp . .

# Create necessary directories
RUN mkdir -p data logs && \
    chown -R whatsapp:whatsapp /app

# Switch to non-root user
USER whatsapp

# Expose port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:7860/ || exit 1

# Default command
CMD ["python", "app.py"]

# Production-optimized stage with additional security
FROM production as production-secure

# Switch back to root for security hardening
USER root

# Remove package manager and other sensitive tools
RUN apt-get remove -y apt-get && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Set additional security flags
ENV PYTHONHASHSEED=random

# Switch back to app user
USER whatsapp

# Add metadata
LABEL maintainer="WhatsApp RAG ES Team" \
      description="WhatsApp RAG system with Spanish language support" \
      version="1.0.0" \
      org.opencontainers.image.source="https://github.com/your-org/whatsapp-rag-es"
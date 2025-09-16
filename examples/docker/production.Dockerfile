# Multi-stage production Dockerfile with optimizations
FROM python:3.11-slim as builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies to local user space
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Production stage
FROM python:3.11-slim as production

# Set environment variables for production
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860
ENV LOG_LEVEL=INFO
ENV WORKERS=1

WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    curl \
    tini \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy Python packages from builder stage
COPY --from=builder /root/.local /root/.local

# Make sure scripts in .local are usable
ENV PATH=/root/.local/bin:$PATH

# Create non-root user for security
RUN adduser --disabled-password --gecos '' --uid 1001 appuser

# Copy application code with proper ownership
COPY --chown=appuser:appuser app.py .
COPY --chown=appuser:appuser rag/ ./rag/

# Create necessary directories with proper permissions
RUN mkdir -p logs cache indices data \
    && chown -R appuser:appuser logs cache indices data

# Switch to non-root user
USER appuser

# Health check with better start period for production
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=5 \
    CMD curl -f http://localhost:7860/ || exit 1

# Expose port
EXPOSE 7860

# Use tini as entrypoint for proper signal handling
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["python", "-u", "app.py"]
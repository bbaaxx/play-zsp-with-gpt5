FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860

# Create non-root user
RUN useradd --create-home --shell /bin/bash app

# Set work directory
WORKDIR /home/app

# Install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY --chown=app:app . .

# Create necessary directories
RUN mkdir -p data logs \
    && chown -R app:app data logs

# Switch to non-root user
USER app

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:7860/ || exit 1

# Expose port
EXPOSE 7860

# Run the application
CMD ["python", "app.py"]
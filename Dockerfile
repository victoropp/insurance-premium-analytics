# Videbimus AI - Insurance Premium Analytics Platform
# Production Docker Container

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Expose port
EXPOSE 8050

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DASH_DEBUG=False

# Run the application with gunicorn
CMD ["gunicorn", "dashboard:server", "-b", "0.0.0.0:8050", "--workers", "4", "--timeout", "120"]
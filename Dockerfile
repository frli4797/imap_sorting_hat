# ============================
# 1. Base builder image
# ============================
FROM python:3.13-slim AS builder

WORKDIR /app

# Install build dependencies if needed by your deps (optional)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy project metadata first for better caching
COPY pyproject.toml ./
COPY README.md .
COPY LICENSE .

# Copy application code
COPY src/ src/

# Cipy requirements file
COPY requirements.txt .

# Install the package (no dev deps)
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install .

# ============================
# 2. Runtime image
# ============================
FROM python:3.13-slim

WORKDIR /app

# Copy installed site-packages from builder
COPY --from=builder /usr/local /usr/local

# Default command — adjust to your app’s entrypoint
CMD ["python", "-m", "ish.app"]
# Stage 1: Builder

FROM python:3.12-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc curl \
    && rm -rf /var/lib/apt/lists/*

# Create virtualenv
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install prod dependencies
COPY requirements/prod.txt .
RUN pip install --no-cache-dir -r prod.txt

# Copy warmup script
COPY scripts/warmup_models.py .

# Download models into image layer
RUN python warmup_models.py


# Stage 2: Runtime

FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy virtualenv from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy model cache
COPY --from=builder /root/.cache /root/.cache

# Copy app source
COPY src/ ./src/
COPY params.yaml ./params.yaml

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

EXPOSE 8000

CMD ["sh", "-c", "uvicorn src.api.app:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
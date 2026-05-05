FROM python:3.11-slim

WORKDIR /app

# Install system dependencies needed for PostgreSQL (asyncpg) and machine learning libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install Python dependencies WITH caching and a longer timeout to prevent network drops
RUN pip install --default-timeout=100 -r requirements.txt

COPY . .

# Run the sweep strategy
CMD ["python", "main.py"]
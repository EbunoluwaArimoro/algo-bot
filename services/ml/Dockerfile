FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install ML dependencies (xgboost, pandas, etc.)
RUN pip install --default-timeout=100 -r requirements.txt

COPY . .

# Keep container alive if you want to run scripts manually, or run a specific script
CMD ["tail", "-f", "/dev/null"]
FROM python:3.11-slim

WORKDIR /app

# Install gcc in case any data libraries (like pandas or ta) need compiling
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install Python dependencies WITH caching and a longer timeout to prevent network drops
RUN pip install --default-timeout=100 -r requirements.txt

COPY . .

# Run the strategy service
CMD ["python", "main.py"]
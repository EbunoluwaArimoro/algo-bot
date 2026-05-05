FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for Postgres and Math libraries
RUN apt-get update && apt-get install -y --no-install-recommends gcc libpq-dev && rm -rf /var/lib/apt/lists/*

# Copy the requirements and install them
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy your trusted code
COPY . .

CMD ["python", "main.py"]
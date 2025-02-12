FROM python:3.11.2-slim-buster

WORKDIR /app

# Install system dependencies (including awscli)
RUN apt-get update && \
    apt-get install -y --no-install-recommends awscli && \
    rm -rf /var/lib/apt/lists/*  # Clean up apt cache

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python3", "app.py"]
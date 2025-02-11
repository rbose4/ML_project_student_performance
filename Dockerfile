# Set a non-root user (important for security)
FROM python:3.8-slim-buster
COPY . /app
WORKDIR /app

RUN apt update -y && apt install awscli -y

RUN pip install -r requirements.txt
CMD ["python3", "app.py"]

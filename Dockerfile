FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies first for better layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project sources.
COPY config ./config
COPY src ./src
COPY main.py .

# Default command runs training mode.
CMD python main.py --mode train && uvicorn main:app --host 0.0.0.0 --port 8080

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY config.yaml .
COPY main.py .
COPY src ./src

RUN mkdir -p /app/artifacts /app/data/raw

EXPOSE 8000

CMD ["python", "main.py", "--config", "config.yaml"]

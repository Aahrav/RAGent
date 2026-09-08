# Deployment Guide

## Local Development
```bash
docker compose up -d   # starts Qdrant + Redis
pip install -r requirements.txt
python src/main.py     # http://localhost:8000
```

## Docker Build
```bash
docker build -t ragent .
docker run -p 8000:8000 --env-file .env ragent
```

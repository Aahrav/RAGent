# Deployment Guide

## Local Development
`ash
docker compose up -d   # starts Qdrant + Redis
pip install -r requirements.txt
python src/main.py     # http://localhost:8000
`

## Docker Build
`ash
docker build -t ragent .
docker run -p 8000:8000 --env-file .env ragent
`
"@ | Set-Content docs\deployment.md -Encoding UTF8

# scripts/
@"
"""CLI script for bulk document ingestion."""
# TODO: Phase 1
# Usage: python scripts/ingest_docs.py --source data/

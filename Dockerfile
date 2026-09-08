FROM python:3.11-slim AS builder
# Pull the incredibly fast 'uv' binary directly from Astral
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Copy only the dependency definitions first (for perfect layer caching)
COPY pyproject.toml uv.lock ./

# Install dependencies into a virtual environment
RUN uv sync --frozen --no-dev --no-install-project

FROM python:3.11-slim AS runtime
WORKDIR /app

# Copy the pre-built virtual environment from the builder
COPY --from=builder /app/.venv /app/.venv
COPY src/ ./src/

EXPOSE 8000

# Execute uvicorn using the Python interpreter inside the virtual environment
CMD ["/app/.venv/bin/python", "-m", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]

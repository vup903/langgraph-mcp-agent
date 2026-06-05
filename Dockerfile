# syntax=docker/dockerfile:1
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install the package (deps + src layout) with good layer caching.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install -e .

COPY . .

# Runs the demo agent (needs an LLM key passed at runtime via --env-file).
ENTRYPOINT ["python", "main.py"]

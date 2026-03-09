FROM python:3.12-slim

WORKDIR /app

# Install UV
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy project files
COPY pyproject.toml uv.lock ./
COPY src/ src/
COPY config/ config/
COPY alembic/ alembic/
COPY alembic.ini .

# Install dependencies
RUN uv sync --frozen --no-dev

# Set Python path
ENV PYTHONPATH=/app/src

CMD ["uv", "run", "python", "-m", "brand_conscience.cli", "health"]

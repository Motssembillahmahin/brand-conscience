# Brand Conscience — Local Development

## Prerequisites

- Python 3.12+
- [UV](https://docs.astral.sh/uv/) (package manager)
- Docker and Docker Compose
- Git

## Quick Start

### 1. Clone and install dependencies

```bash
git clone <repo-url>
cd brand-conscience
uv sync
```

### 2. Set up environment

```bash
cp .env.example .env
# Edit .env with your API keys (see docs/api-keys.md)
```

### 3. Start infrastructure

```bash
docker compose up -d postgres redis opik
```

### 4. Run database migrations

```bash
uv run alembic upgrade head
```

### 5. Verify setup

```bash
uv run brand-conscience health
```

This checks connectivity to PostgreSQL, Redis, and all configured APIs.

## Docker Compose Services

Start everything:
```bash
docker compose up -d
```

Start only infrastructure (for local development):
```bash
docker compose up -d postgres redis opik
```

| Service | Port | URL |
|---------|------|-----|
| PostgreSQL | 5432 | `postgresql://localhost:5432/brand_conscience` |
| Redis | 6379 | `redis://localhost:6379/0` |
| OPIK UI | 5173 | `http://localhost:5173` |
| OPIK API | 8080 | `http://localhost:8080` |

## Running Tests

### Unit tests
```bash
uv run pytest tests/unit/ -v
```

### Integration tests (requires Docker services)
```bash
uv run pytest tests/integration/ -v
```

### E2E tests (requires all services + mocked external APIs)
```bash
uv run pytest tests/e2e/ -v
```

### With coverage
```bash
uv run pytest --cov=brand_conscience --cov-report=html
```

## Common UV Commands

```bash
# Add a dependency
uv add <package>

# Add a dev dependency
uv add --dev <package>

# Sync dependencies
uv sync

# Run a command in the project environment
uv run <command>

# Update lock file
uv lock
```

## Running Workers

### Celery worker
```bash
uv run celery -A brand_conscience.celery_app worker -l info -c 4
```

### Celery beat (scheduler)
```bash
uv run celery -A brand_conscience.celery_app beat -l info
```

## CLI Commands

```bash
# Health check
uv run brand-conscience health

# Force monitoring run
uv run brand-conscience monitor --force

# View campaign status
uv run brand-conscience campaigns list

# Manually approve a campaign
uv run brand-conscience campaigns approve <campaign-id>
```

## Code Quality

```bash
# Lint
uv run ruff check .

# Format
uv run ruff format .

# Type check
uv run mypy src/
```

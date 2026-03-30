.PHONY: help install sync lint format typecheck test test-unit test-integration test-e2e \
       coverage health monitor pipeline db-migrate db-seed db-shell gen-scorer-data merge-scorer-data \
       train-scorer test-scorer train-scorer-clip test-scorer-clip \
       infra-up infra-down docker-up docker-down docker-build \
       worker beat clean logs

# ─── Help ────────────────────────────────────────────────────────────────────

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ─── Setup ───────────────────────────────────────────────────────────────────

install: ## Install all dependencies (runtime + dev)
	uv sync

sync: ## Sync dependencies from lock file
	uv sync --frozen

env: ## Create .env from example
	cp -n .env.example .env || true
	@echo ".env file ready — edit it with your API keys"

# ─── Code Quality ───────────────────────────────────────────────────────────

lint: ## Run ruff linter
	uv run ruff check .

lint-fix: ## Run ruff linter with auto-fix
	uv run ruff check --fix .

format: ## Run ruff formatter
	uv run ruff format .

format-check: ## Check formatting without changing files
	uv run ruff format --check .

typecheck: ## Run mypy type checker
	uv run mypy src/ --ignore-missing-imports

check: lint format-check typecheck ## Run all code quality checks

# ─── Testing ────────────────────────────────────────────────────────────────

test: ## Run all tests
	uv run pytest tests/ -v

test-unit: ## Run unit tests only
	uv run pytest tests/unit/ -v

test-integration: ## Run integration tests (requires infra)
	uv run pytest tests/integration/ -v -m integration

test-e2e: ## Run end-to-end tests
	uv run pytest tests/e2e/ -v -m e2e

coverage: ## Run tests with HTML coverage report
	uv run pytest tests/ --cov=brand_conscience --cov-report=html --cov-report=term
	@echo "Coverage report: htmlcov/index.html"

# ─── Application ────────────────────────────────────────────────────────────

health: ## Run health check against all services
	uv run brand-conscience health

monitor: ## Run a monitoring cycle
	uv run brand-conscience monitor --force

pipeline: ## Run the full autonomous pipeline
	uv run brand-conscience pipeline

campaigns: ## List all campaigns
	uv run brand-conscience campaigns list

# ─── Database ───────────────────────────────────────────────────────────────

db-migrate: ## Run database migrations
	uv run alembic upgrade head

db-rollback: ## Rollback last migration
	uv run alembic downgrade -1

db-seed: ## Seed database with sample data
	uv run python scripts/seed_db.py

db-reset: db-rollback db-migrate db-seed ## Reset database (rollback + migrate + seed)

db-shell: ## Open psql shell to the database
	docker compose exec postgres psql -U brand_conscience -d brand_conscience

# ─── Infrastructure (local dev — DB/Redis/OPIK only) ───────────────────────

infra-up: ## Start infrastructure services (postgres, redis, opik)
	docker compose up -d postgres redis opik

infra-up-lite: ## Start postgres + redis only (skip opik if image pull fails)
	docker compose up -d postgres redis

infra-down: ## Stop infrastructure services
	docker compose down

infra-logs: ## Tail infrastructure logs
	docker compose logs -f postgres redis opik

# ─── Docker (full stack) ────────────────────────────────────────────────────

docker-build: ## Build application Docker image
	docker compose build

docker-up: ## Start all services (infra + app + workers)
	docker compose up -d

docker-down: ## Stop all services and remove volumes
	docker compose down -v

docker-logs: ## Tail all service logs
	docker compose logs -f

# ─── Celery Workers ─────────────────────────────────────────────────────────

worker: ## Start Celery worker (local)
	uv run celery -A brand_conscience.celery_app worker -l info -c 4

beat: ## Start Celery beat scheduler (local)
	uv run celery -A brand_conscience.celery_app beat -l info

# ─── ML Scripts ─────────────────────────────────────────────────────────────

bootstrap: ## Bootstrap historical ad embeddings
	uv run python scripts/bootstrap_historical.py \
		--ads-dir data/historical_ads \
		--metadata data/metadata.json \
		--output data/embeddings.pt

gen-scorer-data: ## Generate prompt scorer training data via LLM judge
	uv run python scripts/generate_prompt_training_data.py \
		--output data/prompt_performance.json \
		--n-samples 200

merge-scorer-data: ## Merge all prompt_performance*.json into one training file
	uv run python -c "\
	import json, glob; \
	files = sorted(glob.glob('data/prompt_performance*.json')); \
	merged = [item for f in files for item in json.load(open(f))]; \
	json.dump(merged, open('data/prompt_performance_merged.json','w'), indent=2); \
	print(f'Merged {len(merged)} samples from {len(files)} files')"

train-scorer: merge-scorer-data ## Train prompt scorer model on all data
	uv run python scripts/train_prompt_scorer.py \
		--data data/prompt_performance_merged.json \
		--output model_checkpoints/prompt_scorer.pt \
		--vocab-output model_checkpoints/prompt_scorer_vocab.json

test-scorer: ## Test prompt scorer by generating images for top/bottom scored prompts
	uv run python scripts/test_prompt_scorer.py \
		--data data/prompt_performance_merged.json \
		--checkpoint model_checkpoints/prompt_scorer.pt \
		--vocab model_checkpoints/prompt_scorer_vocab.json \
		--output-dir test_outputs/prompt_scorer \
		--top-n 5

train-scorer-clip: merge-scorer-data ## Train CLIP-based prompt scorer
	uv run python scripts/train_prompt_scorer.py \
		--data data/prompt_performance_merged.json \
		--output model_checkpoints/prompt_scorer_clip.pt \
		--model-type clip_mlp

test-scorer-clip: ## Test CLIP prompt scorer with image generation
	uv run python scripts/test_prompt_scorer.py \
		--data data/prompt_performance_merged.json \
		--checkpoint model_checkpoints/prompt_scorer_clip.pt \
		--output-dir test_outputs/prompt_scorer_clip \
		--model-type clip_mlp \
		--top-n 5

train-classifier: ## Train quality classifier model
	uv run python scripts/train_quality_classifier.py \
		--embeddings data/embeddings.pt \
		--output model_checkpoints/quality_classifier.pt

# ─── Cleanup ────────────────────────────────────────────────────────────────

clean: ## Remove build artifacts and caches
	rm -rf build/ dist/ *.egg-info .eggs/
	rm -rf .pytest_cache/ .mypy_cache/ .ruff_cache/
	rm -rf htmlcov/ .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@echo "Cleaned."

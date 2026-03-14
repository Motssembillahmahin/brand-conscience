# Brand Conscience

**Fully autonomous Meta advertisement system** that monitors business health and cultural trends, writes its own campaign briefs, generates Gemini-powered ad creatives, deploys via Meta Marketing API, and continuously self-improves through reinforcement learning — all without human intervention unless spend exceeds a configurable threshold.

## What It Does

Brand Conscience operates as a closed-loop advertising engine:

1. **Monitors** business metrics (revenue, inventory, CRM), cultural signals (social trends, news), and creative performance (CTR trends, fatigue) on staggered schedules
2. **Decides** which audience to target and how much budget to allocate using a PPO reinforcement learning agent
3. **Generates** ad prompts from templates, scores them with a trained model (transformer or CLIP MLP), and sends the best to Google Gemini for image creation
4. **Evaluates** every generated creative through a 4-gate pipeline: quality classification, brand alignment, originality, and predicted performance
5. **Deploys** approved creatives via Meta Marketing API with Thompson Sampling A/B testing and real-time bid optimization
6. **Learns** from campaign outcomes — collecting metrics, computing RL rewards, detecting model drift, and triggering automatic retraining

Humans set guardrails (spend thresholds, brand safety rules, bid caps). The system handles everything else.

## Architecture

```
Layer 0: Awareness     →  Business (15m) | Cultural (1h) | Creative (4h) → MomentProfile
Layer 1: Strategy      →  PPO Strategic Agent → Audience + Budget decisions (hourly)
Layer 2: Prompts       →  Template Builder → Prompt Scorer Gate (Transformer or CLIP MLP)
Layer 3: Creative      →  Gemini Generation → 4-Gate Evaluation Pipeline
Layer 4: Deployment    →  Meta API → Tactical RL Agent → A/B Testing + Circuit Breaker
Layer 5: Feedback      →  Metrics Collection → Drift Detection → RL Reward → Retraining
```

Each layer is a LangGraph subgraph. The master graph in `app.py` composes them into a single pipeline with PostgreSQL checkpointing for resumability.

## Prerequisites

- **Python 3.12+**
- **[UV](https://docs.astral.sh/uv/)** — package manager (replaces pip/poetry)
- **Docker & Docker Compose** — for infrastructure services (PostgreSQL, Redis, OPIK)

## API Keys

The system requires these credentials in `.env` (see [docs/api-keys.md](docs/api-keys.md) for details):

| Key | Required | Purpose |
|-----|----------|---------|
| `BC_META__ACCESS_TOKEN` | Yes | Meta Marketing API — campaign deployment and metrics |
| `BC_META__APP_ID` / `APP_SECRET` | Yes | Meta app authentication |
| `BC_META__AD_ACCOUNT_ID` | Yes | Target Meta ad account |
| `BC_ANTHROPIC__API_KEY` | Yes | Claude LLM judge for prompt scoring training data |
| `BC_GEMINI__API_KEY` | Yes | Google Gemini image generation |
| `BC_SLACK__BOT_TOKEN` | Optional | Slack notifications and spend approvals |
| `BC_COMET__API_KEY` | Optional | Comet ML experiment tracking for model training |
| `BC_OPIK__API_KEY` | Optional | OPIK decision tracing (falls back to local) |

## Quick Start

```bash
# Install dependencies
make install

# Set up environment variables
make env
# Edit .env with your API keys

# Start infrastructure (PostgreSQL, Redis, OPIK)
make infra-up

# Run database migrations
make db-migrate

# Verify everything works
make health

# Run the full pipeline
make pipeline
```

## Configuration

All settings live in `config/settings.yaml` with environment-specific overrides via `config/settings.{env}.yaml`. Environment variables prefixed with `BC_` override YAML values (e.g. `BC_META__ACCESS_TOKEN`).

Key configuration options:

| Setting | Default | Description |
|---------|---------|-------------|
| `models.prompt_scorer.type` | `"transformer"` | Scorer architecture: `"transformer"` or `"clip_mlp"` |
| `prompts.scorer_threshold` | `0.7` | Minimum score for a prompt to pass the scoring gate |
| `safety.circuit_breaker.velocity_multiplier` | `3.0` | Spend velocity that triggers circuit breaker |
| `deployment.spend_approval_threshold` | `1000.0` | Daily spend ($) above which human approval is required |
| `drift.psi_threshold` | `0.2` | PSI score that triggers automatic model retraining |

## Model Training

Models must be trained before the pipeline can score prompts and classify creatives. Training data generation and training are handled via Makefile targets.

### Prompt Scorer

Two architectures are available. Both are trained on the same data (prompts scored by a Claude LLM judge) and share the same scoring gate interface — switch between them via `models.prompt_scorer.type` in config.

```bash
# Generate training data (requires Anthropic API key)
make gen-scorer-data

# Option A: Train transformer scorer (from-scratch word-level model)
make train-scorer

# Option B: Train CLIP MLP scorer (pretrained CLIP embeddings + MLP head)
make train-scorer-clip

# Visually validate either scorer by generating images for top/bottom prompts
make test-scorer          # transformer
make test-scorer-clip     # CLIP MLP
```

The CLIP MLP scorer leverages pretrained semantic understanding from OpenCLIP (2B image-text pairs), so it performs better with small training sets (~800 samples). The transformer scorer trains from scratch and may need more data to match.

### Quality Classifier

```bash
# Bootstrap CLIP embeddings from historical ads
make bootstrap

# Train quality classifier
make train-classifier
```

### RL Agents

Strategic and tactical RL agents train online during system operation. Initial policy uses random exploration with safety constraints (circuit breaker, bid caps).

## Use Cases

### Autonomous Campaign Launch
Revenue drops 20% in electronics → system detects it within 15 minutes → strategic agent selects retargeting audience with $500/day budget → generates promotional creatives → deploys A/B test on Meta → tactical agent optimizes bids every 5 minutes → revenue stabilizes without human involvement.

### Circuit Breaker Protection
Spend velocity hits 3x the daily rate in 2 hours → all campaigns pause immediately → Slack alert fires → system enters 1-hour cooldown → re-evaluates at reduced budget. Maximum financial exposure is hard-capped.

### Creative Fatigue Recovery
CTR declines 30% over 10 days → creative monitor flags fatigue → fresh prompts generated from different templates → diversity gate ensures new creatives look different → Thompson Sampling tests old vs new → old creatives retired once new ones prove superior.

### Cultural Sensitivity Response
Trending negative event matches a topic used in active campaigns → brand safety classifier flags it → affected campaigns pause → new creatives generated excluding the sensitive topic → system resumes after brand review.

### Automatic Model Retraining
Drift detector finds PSI > 0.2 on prompt scorer feature distribution → retraining job pulls last 30 days of data → new model evaluated on holdout set → promoted if improved → Slack notification with old vs new metrics.

## Safety Rails

- **Circuit breaker** — pauses all campaigns if spend velocity exceeds 3x daily rate
- **Brand safety classifier** — CLIP embedding similarity screening against risk topics
- **Max bid cap** — hard ceiling at 5x target CPC, warning at 2x
- **Creative diversity enforcer** — minimum CLIP embedding distance between active creatives
- **Spend approval threshold** — campaigns above $1,000/day require human `/approve` via Slack
- **Model quality gates** — policy entropy monitoring, holdout accuracy checks, drift detection

## Makefile Commands

Run `make help` for the full list.

### Setup & Quality

| Command | Description |
|---------|-------------|
| `make install` | Install all dependencies via UV |
| `make env` | Create `.env` from example template |
| `make check` | Run linter + formatter + type checker |
| `make lint` | Run ruff linter |
| `make format` | Run ruff formatter |
| `make typecheck` | Run mypy type checker |

### Testing

| Command | Description |
|---------|-------------|
| `make test-unit` | Run unit tests |
| `make test` | Run all tests |
| `make coverage` | Tests with HTML coverage report |

### ML Training

| Command | Description |
|---------|-------------|
| `make gen-scorer-data` | Generate prompt scorer training data via LLM judge |
| `make merge-scorer-data` | Merge all prompt performance JSON files |
| `make train-scorer` | Train transformer prompt scorer |
| `make train-scorer-clip` | Train CLIP MLP prompt scorer |
| `make test-scorer` | Test transformer scorer with image generation |
| `make test-scorer-clip` | Test CLIP scorer with image generation |
| `make bootstrap` | Bootstrap historical ad embeddings |
| `make train-classifier` | Train quality classifier |

### Infrastructure & Operations

| Command | Description |
|---------|-------------|
| `make infra-up` | Start PostgreSQL, Redis, OPIK |
| `make infra-down` | Stop infrastructure services |
| `make db-migrate` | Run Alembic migrations |
| `make db-seed` | Seed database with sample data |
| `make health` | Check connectivity to all services |
| `make monitor` | Run a monitoring cycle |
| `make pipeline` | Run the full autonomous pipeline |
| `make worker` | Start Celery worker |
| `make beat` | Start Celery beat scheduler |
| `make docker-up` | Start everything via Docker Compose |
| `make docker-down` | Stop all services and remove volumes |
| `make clean` | Remove build artifacts and caches |

## Tech Stack

- **Python 3.12+** with full type annotations
- **LangGraph** — pipeline orchestration with PostgreSQL checkpointing
- **PyTorch** — RL agents (PPO), prompt scorer (transformer or CLIP MLP), quality classifier (MLP)
- **OpenCLIP** (ViT-L-14) — image/text embeddings for quality, brand alignment, and diversity
- **Google Gemini** — AI image generation
- **Meta Marketing API** — campaign deployment and metrics
- **Celery + Redis** — task scheduling and async processing
- **PostgreSQL** (pgvector) — state persistence, campaign data, model checkpoints
- **OPIK** — end-to-end decision tracing
- **structlog** — structured JSON logging
- **Comet ML** — experiment tracking for model training
- **UV** — dependency management
- **Ruff** — linting and formatting

## Project Structure

```
src/brand_conscience/
├── app.py                    # Master LangGraph pipeline
├── celery_app.py             # Celery factory + beat schedule
├── cli.py                    # CLI entry points
├── common/                   # Config, database, logging, tracing, notifications
├── models/                   # ML models (CLIP, prompt scorer, quality classifier, RL, safety)
├── db/                       # ORM tables and query functions
├── layer0_awareness/         # Signal monitoring and MomentProfile
├── layer1_strategy/          # Strategic RL agent, audience, budget
├── layer2_prompts/           # Prompt templates, builder, scoring gate
├── layer3_creative/          # Gemini client, 4-gate evaluation, asset manager
├── layer4_deployment/        # Meta client, campaign manager, tactical RL, A/B testing
└── layer5_feedback/          # Metrics collection, drift detection, retraining
```

## Development

```bash
# Run all code quality checks
make check

# Run unit tests
make test-unit

# Run tests with coverage
make coverage
```

Code style is enforced by **Ruff** (line length: 100). All code is fully typed (Python 3.12+ annotations). Logging uses **structlog** (structured JSON). Tracing uses **OPIK** at every layer boundary.

## Documentation

Detailed docs are in the `docs/` directory:

- [Overview](docs/overview.md) — vision and philosophy
- [Architecture](docs/architecture.md) — 6-layer system design and data flow
- [Use Cases](docs/use-cases.md) — detailed scenario walkthroughs
- [Workflow](docs/workflow.md) — end-to-end pipeline sequence
- [Models](docs/models.md) — ML model specifications
- [Monitoring](docs/monitoring.md) — three nervous systems
- [Deployment](docs/deployment.md) — Meta API integration and A/B testing
- [Safety](docs/safety.md) — circuit breaker, brand safety, bid caps
- [Observability](docs/observability.md) — OPIK tracing and structlog config
- [API Keys](docs/api-keys.md) — required credentials
- [Local Dev](docs/local-dev.md) — development setup guide

## License

Proprietary. All rights reserved.

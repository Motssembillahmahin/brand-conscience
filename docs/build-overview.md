# Brand Conscience — Build Overview

## Build Phases

### Phase 0: Project Setup & Docs
- Project configuration (CLAUDE.md, memory)
- All documentation files
- Foundation for development workflow

### Phase 1: Foundation
- Python project setup (pyproject.toml, UV)
- Package structure with all __init__.py files
- Common utilities: config, database, logging, tracing, notifications
- Database schema and Alembic migrations
- Docker Compose infrastructure
- Test fixtures and factories

### Phase 2: ML Models
- CLIP encoder wrapper (OpenCLIP ViT-L-14)
- Quality classifier (CLIP → 4-class MLP)
- Prompt scorer (transformer regressor, 4 layers, 256d, 4 heads)
- RL networks (PPO with actor-critic for strategic + tactical agents)
- Safety models (brand safety classifier, diversity enforcer)
- Training and bootstrap scripts

### Phase 3: Monitoring + Feedback (Edges)
- Layer 0: Signal dataclasses, monitors, moment profile aggregation
- Layer 5: Metrics collection, drift detection, model updater, reporter
- Celery periodic tasks for both layers

### Phase 4: Strategy + Prompts
- Layer 1: Strategic state, audience selector, budget allocator, strategic RL agent
- Layer 2: Prompt templates, builder, scoring gate
- LangGraph subgraphs for both layers

### Phase 5: Creative + Deployment
- Layer 3: Gemini client, 4-gate evaluation pipeline, asset manager
- Layer 4: Meta client, campaign manager, tactical RL agent, A/B testing, circuit breaker
- LangGraph subgraphs for both layers

### Phase 6: Integration
- Master LangGraph composition (app.py)
- Celery app factory with beat schedule
- CLI entry points
- Database query functions
- Integration and E2E tests
- CI/CD pipeline

## Dependency Graph

```
Phase 0 (Docs)
    │
    ▼
Phase 1 (Foundation)
    │
    ├──────────────┐
    ▼              ▼
Phase 2         Phase 3*
(ML Models)     (L0 + L5)
    │              │
    ├──────┬───────┘
    ▼      ▼
Phase 4    Phase 5
(L1+L2)    (L3+L4)
    │         │
    └────┬────┘
         ▼
    Phase 6
  (Integration)
```

*Phase 3 depends on Phase 1 for database/config but not on Phase 2 models directly. However, Phase 5 (Layer 5 feedback) uses models for drift detection, so Phase 2 should be completed first.

## Implementation Order Rationale

1. **Foundation first** — Everything depends on config, database, and logging
2. **Models before layers** — Layers 1-4 consume model outputs; models must exist first
3. **Edges before core** — Layer 0 (input) and Layer 5 (feedback) are the system boundaries; building them early validates the data contracts
4. **Strategy before creative** — Layer 3 and 4 need Layer 1's strategic decisions as input
5. **Integration last** — The master graph and CLI compose all layers after they're individually tested

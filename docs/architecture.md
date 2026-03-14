# Brand Conscience — Architecture

## 6-Layer System Design

Brand Conscience is organized into six layers, each responsible for a distinct phase of the autonomous advertising pipeline. Data flows downward through the layers, and feedback flows upward.

```
┌─────────────────────────────────────────────────────┐
│  Layer 0: Awareness                                  │
│  Business Monitor (15m) │ Cultural Monitor (1h)      │
│  Creative Monitor (4h)  │ → MomentProfile            │
├─────────────────────────────────────────────────────┤
│  Layer 1: Strategy                                   │
│  Strategic RL Agent (PPO) → Audience + Budget         │
├─────────────────────────────────────────────────────┤
│  Layer 2: Prompt Engineering                         │
│  Template Builder → Prompt Scorer Gate (Transformer   │
│  or CLIP MLP, config-selectable)                     │
├─────────────────────────────────────────────────────┤
│  Layer 3: Creative Production                        │
│  Gemini Generation → 4-Gate Evaluation Pipeline      │
│  (Quality → Brand → Originality → Performance)       │
├─────────────────────────────────────────────────────┤
│  Layer 4: Deployment                                 │
│  Meta API → Tactical RL Agent → A/B Testing          │
│  Campaign State Machine + Circuit Breaker            │
├─────────────────────────────────────────────────────┤
│  Layer 5: Feedback & Learning                        │
│  Metrics Collection → Drift Detection → Retraining   │
│  RL Reward Computation → Policy Updates              │
└─────────────────────────────────────────────────────┘
```

## Data Flow

### Forward Path (Signal → Action)
1. **Layer 0** aggregates signals into a `MomentProfile` with an urgency score
2. **Layer 1** receives the profile; the strategic RL agent selects audience segments and allocates budget
3. **Layer 2** fills prompt templates with strategic context and scores them; only high-scoring prompts proceed
4. **Layer 3** sends prompts to Gemini for image generation, then runs the 4-gate evaluation pipeline
5. **Layer 4** deploys approved creatives via Meta Marketing API with tactical RL-optimized bids and placements
6. **Layer 5** collects performance metrics from Meta and computes RL rewards

### Feedback Path (Outcome → Learning)
1. **Layer 5** computes strategic and tactical rewards from actual campaign metrics
2. Strategic rewards feed back to **Layer 1** (hourly policy updates)
3. Tactical rewards feed back to **Layer 4** (per-minute bid adjustments)
4. Drift detection in **Layer 5** can trigger retraining of prompt scorer and quality classifier
5. Circuit breaker in **Layer 4** can pause campaigns and notify humans

## Orchestration

- **LangGraph** — Each layer is a LangGraph subgraph; the master graph in `app.py` composes them
- **Celery + Redis** — Periodic monitoring tasks and async processing
- **PostgreSQL** — State persistence, campaign data, LangGraph checkpoints
- **OPIK** — End-to-end trace threading across all layers

## State Management

All state is persisted in PostgreSQL:
- Campaign lifecycle state machine (DRAFT → PENDING_APPROVAL → LIVE → PAUSED → COMPLETED)
- LangGraph checkpoint state for resumability
- RL model checkpoints and training history
- Moment profiles and signal history for drift detection

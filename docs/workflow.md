# Brand Conscience — End-to-End Workflow

## Pipeline Execution Sequence

### Step 1: Signal Collection (Layer 0)

Three monitors run on staggered schedules:

```
Business Monitor (every 15 min)
  → Revenue data, inventory levels, CRM events
  → Outputs: BusinessSignal[]

Cultural Monitor (every 1 hour)
  → Social trend APIs, news sentiment, hashtag velocity
  → Brand safety filter applied to each signal
  → Outputs: CulturalSignal[]

Creative Monitor (every 4 hours)
  → Ad fatigue metrics, competitor creative shifts
  → Outputs: CreativeSignal[]
```

### Step 2: Moment Profile Assembly (Layer 0)

All signals are aggregated into a `MomentProfile`:
- Normalized urgency score (0.0–1.0)
- Signal category breakdown
- Recommended action type (launch, adjust, pause, refresh)
- Affected product categories and audience segments

### Step 3: Strategic Decision (Layer 1)

The strategic RL agent receives the MomentProfile:
- **State vector**: moment features + current portfolio state + budget remaining
- **Action space**: audience segment selection + budget allocation
- **Output**: `StrategicDecision` with target audiences, daily budget, campaign objective

Runs hourly or on-demand when urgency > 0.7.

### Step 4: Prompt Construction (Layer 2)

Prompt builder fills templates with strategic context:
- Product/service details from catalog
- Target audience characteristics
- Campaign objective and tone
- Seasonal/cultural context from signals

Each prompt is scored by the prompt scorer model (transformer or CLIP MLP, selected via `models.prompt_scorer.type` in config). Only prompts scoring above threshold (configurable, default 0.7) proceed to creative generation.

### Step 5: Creative Generation & Evaluation (Layer 3)

For each approved prompt:
1. **Gemini generation** — 3-5 image variants per prompt
2. **Gate 1: Quality** — CLIP quality classifier rejects low-quality images
3. **Gate 2: Brand Alignment** — CLIP cosine similarity against brand reference embeddings
4. **Gate 3: Originality** — Diversity enforcer ensures minimum distance from existing creatives
5. **Gate 4: Performance Prediction** — Predicted CTR/engagement from historical patterns

Creatives passing all 4 gates are stored as deployment candidates.

### Step 6: Campaign Deployment (Layer 4)

Campaign manager orchestrates Meta API deployment:
1. Create campaign structure (Campaign → AdSet → Ad)
2. Configure A/B test groups with Thompson Sampling
3. Set initial bids from tactical RL agent
4. Apply circuit breaker constraints (max bid, spend velocity limits)
5. Transition state: DRAFT → PENDING_APPROVAL → LIVE

For campaigns below spend threshold, PENDING_APPROVAL is auto-approved.

### Step 7: Real-Time Optimization (Layer 4)

Tactical RL agent runs every few minutes on live campaigns:
- Adjusts bids per placement
- Shifts budget between ad sets
- Pauses underperforming ads
- Circuit breaker monitors spend velocity

### Step 8: Feedback & Learning (Layer 5)

Continuous feedback loop:
1. **Metrics collection** — CTR, CPC, ROAS, impressions from Meta API
2. **Reward computation** — Strategic reward (ROAS, audience quality) + tactical reward (CPC efficiency)
3. **Policy updates** — PPO updates for both RL agents
4. **Drift detection** — PSI/KL divergence monitoring on all model inputs
5. **Reporting** — Daily Slack summary with key metrics and decisions made

## Trace Threading

Every pipeline execution is traced end-to-end via OPIK:
```
MomentProfile → StrategicDecision → PromptConstruction → GeminiCall → MetaDeploy → ActualMetrics → RLReward → PolicyUpdate
```

Each step is a span in the parent trace, enabling full auditability of every autonomous decision.

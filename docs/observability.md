# Brand Conscience — Observability

## OPIK Tracing Strategy

### Trace Hierarchy

Every pipeline run gets one parent trace threading through all layers:

```
MomentProfile #MP-4821
  → StrategicDecision #SD-4821 (state vector, action, audience, budget)
    → PromptConstruction #PC-4821-01..03 (template, score, pass/fail)
      → GeminiCall #GC-4821-01 → REJECTED (brand gate: 0.43)
      → GeminiCall #GC-4821-02 → PASSED (predicted CTR: 3.2%)
        → MetaDeploy #MD-4821 → LIVE
          → ActualCTR: 2.8% → PredictionError: -12.5%
            → RLReward: 0.72 → PolicyUpdate #PU-4821
```

### Integration Points

- **LangGraph**: `track_langgraph(graph, OpikTracer(...))` at each subgraph
- **Cross-layer**: Trace propagation via `opik_context.get_distributed_trace_headers()` in Celery task headers
- **Celery tasks**: `task_prerun` signal extracts trace headers and continues the span

### What Gets Traced

| Layer | Traced Data |
|-------|-------------|
| L0 | Signal count, urgency score, recommended action |
| L1 | State vector, action probabilities, selected audience, budget |
| L2 | Template name, prompt text, scorer output, pass/fail |
| L3 | Gemini request/response, gate scores (all 4), pass/fail per gate |
| L4 | Meta API calls, bid values, campaign state transitions, A/B group assignments |
| L5 | Collected metrics, computed rewards, drift scores, retrain decisions |

### OPIK Dashboard

Available at `http://localhost:5173` in development. Provides:
- Trace timeline view (waterfall of all spans)
- Decision audit (why was this audience/creative/bid chosen?)
- Model performance over time
- Error rate and latency tracking

## structlog Configuration

### Format

- **Production**: JSON output for log aggregation
- **Development**: Colored console renderer for human readability

### Context Binding

Automatic context propagation via Python `contextvars`:
- `trace_id` — OPIK trace ID
- `campaign_id` — active campaign being processed
- `task_id` — Celery task ID (when running in worker)
- `layer` — current layer name (e.g., "layer0_awareness")

### Per-Layer Log Levels

| Layer | Default Level | Rationale |
|-------|--------------|-----------|
| L0 Awareness | INFO | Signal summaries, not raw data |
| L1 Strategy | INFO | Decision summaries |
| L2 Prompts | DEBUG | Useful for prompt engineering iteration |
| L3 Creative | INFO | Generation + gate results |
| L4 Deployment | WARNING | Only errors and circuit breaker events |
| L5 Feedback | INFO | Metric summaries and drift alerts |

### Celery Integration

`task_prerun` signal automatically binds:
- `task_name` and `task_id` to structlog context
- Extracts OPIK trace headers from task kwargs for trace continuation

## Debugging Guide

### Common Investigation Flows

**"Why did this campaign get created?"**
1. Find the MomentProfile trace in OPIK by timestamp
2. Check urgency score and contributing signals
3. Follow trace to StrategicDecision → see audience/budget rationale

**"Why was this creative rejected?"**
1. Find the GeminiCall span in OPIK
2. Check each gate's score in the child spans
3. Identify which gate rejected and the threshold vs. actual value

**"Why did spend spike?"**
1. Check circuit breaker logs (WARNING level in L4)
2. Find tactical agent traces around the spike time
3. Check bid values and placement allocation decisions

**"Why did the model retrain?"**
1. Check drift detector logs in L5
2. Find the drift score that exceeded threshold
3. Review old vs. new model metrics in the retrain trace

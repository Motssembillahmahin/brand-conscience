# Brand Conscience — Monitoring Systems

## Three Nervous Systems

Brand Conscience uses three monitoring systems running at different cadences, each responsible for a different class of signals.

## 1. Business Monitor (every 15 minutes)

**Purpose**: Track core business health metrics that indicate immediate advertising needs.

**Signals collected**:
- Revenue by product category (vs. trailing 7-day average)
- Inventory levels and velocity
- CRM events (new leads, churn signals, LTV changes)
- Conversion funnel metrics (cart abandonment rate, checkout completion)

**Thresholds**:
- Revenue drop >15% triggers elevated urgency
- Inventory surplus >2x normal triggers promotional push
- Churn spike >10% triggers retention campaign consideration

**Data sources**: Business APIs, database queries, webhook events

## 2. Cultural Monitor (every 1 hour)

**Purpose**: Track social and cultural trends that create advertising opportunities or risks.

**Signals collected**:
- Trending topics and hashtags relevant to brand categories
- Sentiment analysis on brand mentions
- Competitor campaign activity and messaging shifts
- News events affecting target markets

**Brand safety filter**: Every cultural signal passes through the brand safety classifier before being included in the MomentProfile. Signals flagged as high-risk are marked for avoidance, not opportunity.

**Data sources**: Social media APIs, news APIs, sentiment analysis services

## 3. Creative Monitor (every 4 hours)

**Purpose**: Track the health and effectiveness of currently deployed ad creatives.

**Signals collected**:
- CTR trend per creative (declining = fatigue)
- Frequency cap saturation (audience overlap)
- Competitor creative style shifts (detected via CLIP embedding clustering)
- Creative diversity score across active campaigns

**Thresholds**:
- CTR decline >30% over 3 days flags creative fatigue
- Frequency >5x per user triggers audience expansion consideration
- Diversity score <0.3 triggers refresh

**Data sources**: Meta Marketing API, CLIP embeddings of competitor ads

## Signal Aggregation → MomentProfile

All signals from the three monitors are combined into a `MomentProfile`:

```
MomentProfile:
  timestamp: datetime
  urgency_score: float (0.0–1.0)
  business_signals: list[BusinessSignal]
  cultural_signals: list[CulturalSignal]
  creative_signals: list[CreativeSignal]
  recommended_action: ActionType  # LAUNCH | ADJUST | PAUSE | REFRESH | HOLD
  affected_categories: list[str]
  affected_audiences: list[str]
  context_summary: str  # LLM-generated natural language summary
```

The urgency score is a weighted combination:
- Business signal severity: 50% weight
- Cultural signal relevance: 30% weight
- Creative signal urgency: 20% weight

Profiles with urgency > 0.7 trigger immediate pipeline execution rather than waiting for the next scheduled run.

## Threshold Triggers

Beyond scheduled monitoring, certain events trigger immediate signal collection:
- Spend velocity anomaly detected by circuit breaker
- External webhook indicating major business event
- Manual trigger via CLI (`brand-conscience monitor --force`)

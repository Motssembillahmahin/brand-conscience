# Brand Conscience — Use Cases

## 1. Autonomous Campaign Launch

**Trigger**: Business monitor detects 20% revenue drop in product category X over last 48 hours.

**Flow**:
1. Layer 0 generates a MomentProfile with high urgency (0.85)
2. Layer 1 strategic agent selects retargeting audience for category X, allocates $500/day budget
3. Layer 2 builds promotional prompts emphasizing value proposition, scores above threshold
4. Layer 3 generates 5 creative variants via Gemini, 3 pass all 4 gates
5. Layer 4 deploys A/B test with 3 variants + holdout group via Meta API
6. Layer 5 collects CTR/ROAS metrics, tactical agent optimizes bids every 5 minutes
7. After 24h, Thompson Sampling converges on best variant, budget shifts accordingly

**Outcome**: Revenue stabilizes without any human involvement.

## 2. Circuit Breaker Trigger

**Trigger**: Spend velocity exceeds 3x the daily budget within first 2 hours.

**Flow**:
1. Layer 4 circuit breaker detects anomalous spend rate
2. All active campaigns in the affected ad account are paused immediately
3. Slack notification sent to ops channel with spend details and campaign IDs
4. System enters cooldown period (configurable, default 1 hour)
5. After cooldown, system re-evaluates and may resume at reduced budget

**Outcome**: Maximum financial exposure is capped. Human is notified but doesn't need to act unless they want to override.

## 3. Drift Detection & Retrain

**Trigger**: Layer 5 drift detector finds PSI > 0.2 on prompt scorer feature distribution.

**Flow**:
1. Drift detector flags the prompt scorer model as degraded
2. Model updater pulls last 30 days of prompt-performance pairs from database
3. Retraining job runs (prompt scorer fine-tune, ~15 minutes on GPU)
4. New model checkpoint is evaluated against holdout set
5. If performance improves, checkpoint is promoted; old model is archived
6. Slack notification with retrain summary (old vs new metrics)

**Outcome**: Model quality stays current without manual ML ops intervention.

## 4. Cultural Sensitivity Response

**Trigger**: Cultural monitor detects trending negative event related to a topic used in active campaigns.

**Flow**:
1. Layer 0 cultural monitor identifies the signal via social API
2. Brand safety classifier flags the topic as high-risk
3. MomentProfile urgency is elevated; affected campaigns are flagged
4. Layer 4 campaign manager pauses affected campaigns
5. Layer 2 regenerates prompts excluding the sensitive topic
6. New creatives go through full 4-gate evaluation before redeployment

**Outcome**: Brand is protected from association with negative events.

## 5. Creative Fatigue Recovery

**Trigger**: Creative monitor detects CTR decline >30% on a campaign running for 10+ days.

**Flow**:
1. Layer 0 flags creative fatigue in MomentProfile
2. Layer 1 maintains same audience/budget but requests creative refresh
3. Layer 2 generates new prompt variants with different angles
4. Layer 3 produces new creatives, diversity gate ensures they differ from fatigued ads
5. Layer 4 deploys new creatives alongside old ones, Thompson Sampling tests
6. Old creatives are retired once new ones prove superior

**Outcome**: Campaign performance recovers without budget interruption.

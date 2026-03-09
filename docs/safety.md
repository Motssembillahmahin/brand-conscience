# Brand Conscience — Safety Rails

## Design Principle

Every autonomous action has at least one safety constraint. The system is designed to fail safe — when in doubt, it pauses rather than spends.

## 1. Circuit Breaker

**Purpose**: Prevent runaway spend from any cause (API errors, RL policy bugs, market anomalies).

**Triggers**:
- Spend velocity exceeds 3x daily budget rate in any 2-hour window
- Single campaign spend exceeds 150% of allocated budget
- Total account spend exceeds daily cap

**Actions**:
1. Immediately pause all campaigns in affected ad account
2. Send Slack alert with details (campaign IDs, spend amounts, trigger reason)
3. Enter cooldown period (default: 1 hour)
4. After cooldown, re-evaluate and optionally resume at 50% budget

**Configuration**: `config/settings.yaml` → `safety.circuit_breaker`

## 2. Brand Safety Classifier

**Purpose**: Prevent ads from being associated with harmful, controversial, or off-brand content.

**How it works**:
- CLIP text embeddings of cultural signals compared against curated risk-topic embeddings
- Cosine similarity > 0.6 flags a signal as risky
- Risky signals are excluded from MomentProfile opportunity assessment
- Active campaigns using related topics are paused for review

**Risk categories**:
- Political controversy
- Natural disasters / tragedies
- Social unrest
- Brand-specific exclusions (configured per deployment)

## 3. Max Bid Cap

**Purpose**: Prevent the tactical RL agent from setting unreasonably high bids.

**Rules**:
- Hard ceiling: No bid can exceed 5x the campaign's target CPC
- Soft ceiling: Bids above 2x target trigger logging + Slack warning
- Per-placement caps: Different maximums for Feed, Stories, Reels, etc.
- Rate limiting: Bid changes limited to once per 5 minutes per ad set

## 4. Creative Diversity Enforcer

**Purpose**: Prevent ad fatigue by ensuring creative variety across campaigns.

**How it works**:
- CLIP image embeddings computed for all active creatives
- New creative must have cosine distance > 0.3 from all active creatives in same campaign
- Cross-campaign diversity: minimum 5 distinct creative clusters across all active campaigns
- Fatigue detection: CTR decline > 30% over 3 days triggers automatic creative refresh

## 5. Spend Threshold Approval

**Purpose**: Require human approval for high-spend campaigns.

**Rules**:
- Daily budget below threshold (default $1,000): auto-approved
- Daily budget above threshold: Slack notification to approvers
- Requires explicit `/approve <campaign_id>` response
- Unapproved campaigns stay in PENDING_APPROVAL state indefinitely

## 6. Model Quality Gates

**Purpose**: Prevent degraded models from making decisions.

**Checks**:
- Prompt scorer must achieve >0.8 correlation on holdout set
- Quality classifier must maintain >85% accuracy on test set
- RL agents: policy entropy must stay above minimum threshold (prevents collapse)
- Drift detector triggers retrain when PSI > 0.2

## Recovery Procedures

| Scenario | Automatic Response | Human Escalation |
|----------|-------------------|------------------|
| Spend spike | Pause + cooldown | Slack alert |
| Brand safety trigger | Pause affected campaigns | Slack alert with context |
| Model drift | Retrain + evaluate | Slack if retrain fails |
| RL policy collapse | Revert to previous checkpoint | Slack alert |
| Meta API errors | Exponential backoff + retry | Slack after 3 failures |
| All circuits tripped | Full system pause | Urgent Slack + email |

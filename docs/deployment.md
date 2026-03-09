# Brand Conscience — Deployment & Meta Integration

## Meta Marketing API Integration

### API Operations

Brand Conscience uses Meta's Marketing API for full campaign lifecycle management:

- **Campaign CRUD**: Create, read, update, delete campaigns
- **AdSet management**: Audience targeting, budget, schedule, placement
- **Ad creation**: Creative upload, ad copy, CTA configuration
- **Insights retrieval**: Performance metrics (CTR, CPC, ROAS, impressions, spend)

### Campaign State Machine

```
DRAFT → PENDING_APPROVAL → LIVE → PAUSED → COMPLETED
                │                    ↑
                │                    │
                └── (auto-approve) ──┘
                     if spend < threshold
```

**States**:
- `DRAFT`: Campaign structure created, not yet submitted
- `PENDING_APPROVAL`: Awaiting approval (auto if below spend threshold, manual Slack notification if above)
- `LIVE`: Active and spending on Meta
- `PAUSED`: Temporarily halted (circuit breaker, manual, or fatigue)
- `COMPLETED`: Budget exhausted or end date reached

### Approval Flow

- **Below threshold** (configurable, default $1,000/day): Auto-approved, transitions directly to LIVE
- **Above threshold**: Slack notification sent to approvers channel with campaign details, budget, and rationale. Requires manual `/approve <campaign_id>` response.

## A/B Testing Strategy

### Thompson Sampling Multi-Armed Bandit

Each campaign deploys multiple creative variants as "arms" in a Thompson Sampling MAB:

1. Each variant starts with a Beta(1, 1) prior (uniform)
2. As impressions and conversions accumulate, posteriors update
3. Budget allocation shifts toward better-performing variants
4. Convergence threshold: 95% probability of best arm identified

### Holdout Groups

- 10% of audience is held out from all variants for baseline measurement
- Holdout sees no ads (pure control)
- Enables true incremental lift measurement

### Attribution Windows

- Click-through: 7 days (default)
- View-through: 1 day (default)
- Configurable per campaign objective

## Tactical Optimization

The tactical RL agent runs every few minutes on live campaigns:

1. **Bid adjustment**: Increase/decrease bids per placement based on real-time performance
2. **Budget reallocation**: Shift daily budget between ad sets
3. **Placement optimization**: Enable/disable specific placements (Feed, Stories, Reels, etc.)
4. **Pacing control**: Ensure daily budget is spent evenly, not front-loaded

All adjustments respect circuit breaker constraints (max bid cap, spend velocity limits).

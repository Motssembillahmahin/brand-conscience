# Brand Conscience — Overview

## Vision

Brand Conscience is a fully autonomous Meta advertisement system that operates without human intervention. It monitors business health, cultural trends, and creative performance to generate its own campaign briefs, produce AI-powered ad creatives, deploy them via Meta's Marketing API, and continuously self-improve through reinforcement learning.

Human oversight is required only when spend exceeds a configurable threshold.

## Philosophy

1. **Autonomous by default** — The system makes all campaign decisions independently, from audience targeting to bid optimization. Humans set guardrails, not instructions.

2. **Signal-driven** — Every campaign decision traces back to a measurable signal: revenue trends, social sentiment, ad fatigue metrics, or competitive shifts.

3. **Safety-first autonomy** — Circuit breakers, brand safety classifiers, bid caps, and creative diversity enforcers ensure the system cannot cause runaway spend or brand damage.

4. **Continuous learning** — Two RL agents (strategic + tactical) optimize at different time horizons. Model drift detection triggers automatic retraining.

5. **Full observability** — Every decision is traced end-to-end via OPIK, from the initial moment profile through creative generation to final campaign performance.

## Core Capabilities

- **Real-time monitoring** of business metrics, cultural signals, and creative performance
- **Autonomous brief generation** based on detected opportunities and threats
- **AI creative production** via Google Gemini with multi-gate quality evaluation
- **Meta campaign deployment** with A/B testing and Thompson Sampling optimization
- **Self-improving RL agents** for both strategic (hourly) and tactical (per-minute) decisions
- **Drift detection** and automatic model retraining when performance degrades

## What This Is Not

- Not a dashboard or reporting tool — it takes action
- Not a recommendation engine — it executes autonomously
- Not a creative tool for humans — it generates and evaluates its own creatives
- Not a rule-based system — it learns optimal strategies from outcomes

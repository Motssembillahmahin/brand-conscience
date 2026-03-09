# Brand Conscience — ML Models

## 1. CLIP Encoder

**Purpose**: Shared image/text embedding backbone used by quality classifier, brand alignment gate, and diversity enforcer.

- **Model**: OpenCLIP ViT-L-14 (pretrained on LAION-2B)
- **Output**: 768-dimensional embeddings
- **Usage**: Frozen weights, inference only
- **Location**: `src/brand_conscience/models/embeddings/clip_encoder.py`

## 2. Prompt Scorer

**Purpose**: Predicts the quality/performance potential of a text prompt before sending it to Gemini for image generation.

- **Architecture**: Transformer regressor
  - 4 layers, 256 hidden dimensions, 4 attention heads
  - Input: tokenized prompt text
  - Output: scalar score (0.0–1.0)
- **Training data**: Historical prompt → ad performance mappings
- **Training**: Supervised regression on CTR/engagement correlation
- **Threshold**: Configurable gate (default 0.7)
- **Location**: `src/brand_conscience/models/prompt_scorer/`

## 3. Quality Classifier

**Purpose**: Classifies generated images into quality tiers before deployment.

- **Architecture**: MLP on CLIP embeddings
  - Input: 768-dim CLIP image embedding
  - Hidden: [512, 256] with ReLU + dropout
  - Output: 4 classes (excellent, good, acceptable, reject)
- **Training data**: Historical ads labeled by performance quartile
- **Gate behavior**: Only "excellent" and "good" pass
- **Location**: `src/brand_conscience/models/quality_classifier/`

## 4. Strategic RL Agent

**Purpose**: Makes hourly decisions about audience targeting and budget allocation.

- **Algorithm**: PPO (Proximal Policy Optimization)
- **Network**: Actor-Critic
  - Shared backbone: [512, 256] MLP
  - Actor head: outputs action probabilities (audience segment + budget level)
  - Critic head: outputs state value estimate
- **State space**: MomentProfile features + portfolio state + budget remaining
- **Action space**: Discrete audience segments × continuous budget allocation
- **Reward**: Weighted combination of ROAS, audience quality score, budget efficiency
- **Update frequency**: Hourly
- **Location**: `src/brand_conscience/models/rl/`

## 5. Tactical RL Agent

**Purpose**: Makes per-minute decisions about bids and placement optimization.

- **Algorithm**: PPO (shared implementation with strategic agent)
- **Network**: Actor-Critic (smaller than strategic)
  - Shared backbone: [256, 128] MLP
  - Actor head: bid adjustments per placement
  - Critic head: state value estimate
- **State space**: Current campaign metrics + bid history + placement performance
- **Action space**: Continuous bid multipliers per placement
- **Reward**: CPC efficiency, delivery pacing, spend velocity compliance
- **Update frequency**: Every few minutes
- **Location**: `src/brand_conscience/models/rl/`

## 6. Brand Safety Classifier

**Purpose**: Screens cultural signals and ad content for brand safety risks.

- **Approach**: CLIP text embeddings + cosine similarity against known-risk topic embeddings
- **Threshold**: Configurable sensitivity (default: flag if similarity > 0.6 to any risk topic)
- **Location**: `src/brand_conscience/models/safety/brand_safety.py`

## 7. Diversity Enforcer

**Purpose**: Ensures new creatives are sufficiently different from existing ones to prevent ad fatigue.

- **Approach**: CLIP image embeddings + minimum pairwise cosine distance
- **Threshold**: New creative must have distance > 0.3 from all active creatives in same campaign
- **Location**: `src/brand_conscience/models/safety/diversity.py`

## Training Pipeline

1. **Bootstrap**: `scripts/bootstrap_historical.py` — CLIP-embeds historical ads and maps to performance
2. **Prompt scorer**: `scripts/train_prompt_scorer.py` — trains transformer regressor
3. **Quality classifier**: `scripts/train_quality_classifier.py` — trains MLP on CLIP embeddings
4. **RL agents**: Train online during system operation; initial policy is random exploration with safety constraints

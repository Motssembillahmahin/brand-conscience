# Brand Conscience — ML Models

## 1. CLIP Encoder

**Purpose**: Shared image/text embedding backbone used by quality classifier, brand alignment gate, and diversity enforcer.

- **Model**: OpenCLIP ViT-L-14 (pretrained on LAION-2B)
- **Output**: 768-dimensional embeddings
- **Usage**: Frozen weights, inference only
- **Location**: `src/brand_conscience/models/embeddings/clip_encoder.py`

## 2. Prompt Scorer

**Purpose**: Predicts the quality/performance potential of a text prompt before sending it to Gemini for image generation. Two architectures are available, selected via `models.prompt_scorer.type` in config.

### 2a. Transformer Scorer (`type: "transformer"`, default)

- **Architecture**: Transformer regressor
  - 4 layers, 256 hidden dimensions, 4 attention heads
  - Input: tokenized prompt text (word-level vocab, trained from scratch)
  - Output: scalar score (0.0–1.0)
- **Checkpoint**: `model_checkpoints/prompt_scorer.pt`
- **Train**: `make train-scorer`
- **Test**: `make test-scorer`

### 2b. CLIP MLP Scorer (`type: "clip_mlp"`)

- **Architecture**: MLP regressor on pretrained CLIP text embeddings
  - CLIP ViT-L-14 encodes prompt → 768-dim embedding (frozen, not trained)
  - Linear(768→256) → ReLU → Dropout(0.1) → Linear(256→64) → ReLU → Dropout(0.1) → Linear(64→1) → Sigmoid
  - Input: 768-dim CLIP text embedding
  - Output: scalar score (0.0–1.0)
- **Advantage**: Leverages pretrained semantic understanding from CLIP (2B image-text pairs), so the MLP only needs to learn the scoring function — works better with small training sets (~800 samples)
- **Checkpoint**: `model_checkpoints/prompt_scorer_clip.pt`
- **Train**: `make train-scorer-clip`
- **Test**: `make test-scorer-clip`

### Common

- **Training data**: Prompts scored by Claude LLM judge on specificity, visual completeness, brand alignment, emotional impact, and actionability (each 0.20 weight)
- **Threshold**: Configurable gate (default 0.7) — shared by both architectures
- **Location**: `src/brand_conscience/models/prompt_scorer/`
- **Comet projects**: `brand-conscience-prompt-scorer` (transformer), `brand-conscience-prompt-scorer-clip` (CLIP MLP)

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
2. **Prompt scorer (transformer)**: `make train-scorer` — trains transformer regressor from scratch
3. **Prompt scorer (CLIP MLP)**: `make train-scorer-clip` — trains MLP on CLIP embeddings
4. **Quality classifier**: `scripts/train_quality_classifier.py` — trains MLP on CLIP embeddings
5. **RL agents**: Train online during system operation; initial policy is random exploration with safety constraints

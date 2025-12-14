# The "G-Factor" Project: Universal Reasoning Boosters

**Status**: 🧪 Research Engine Ready | **Goal**: Universality & Generality

## 1. The Core Hypothesis

We hypothesize the existence of a **"General Reasoning Factor" (G-Vector)** in the activation space of Large Language Models.

While previous research (AFT) successfully identified task-specific steering vectors (e.g., +17% on HellaSwag), these vectors were often brittle and did not transfer well to other tasks (e.g., Math or Logic).

**The G-Factor Hypothesis**: By identifying the *common direction* shared by successful reasoning across multiple domains (Math, Logic, Commonsense), we can construct a single "Universal Booster" that improves performance across the board.

## 2. Research Protocol

### Cycle 1: Generality (The G-Vector)
We train three specialized vectors on **Qwen/Qwen2.5-0.5B**:
1.  $\vec{v}_{math}$ (GSM8K)
2.  $\vec{v}_{logic}$ (ARC-Challenge)
3.  $\vec{v}_{common}$ (HellaSwag)

We then compute the **G-Vector** ($\vec{v}_{G}$) using Vector Fusion techniques (Mean, PCA, or Weighted Average) and test if it improves the *average* accuracy across all three benchmarks.

### Cycle 2: Universality (Cross-Model)
We validate if this methodology transfers to a completely different architecture: **EleutherAI/pythia-410m**. We use **Smart Layer Scouting** to automatically adapt the injection depth to the model's specific topology.

## 3. The Engine: AFT Lab v3.0 (`research_engine.py`)

The research is powered by a unified research engine featuring:
- **Smart Layer Scout**: Automated detection of optimal injection layers (Turbo mode supported).
- **TaskVectorManager**: Management and storage of domain-specific vectors.
- **VectorFusion**: Mathematical operations to synthesize the G-Vector.
- **Turbo Mode**: Rapid iteration using subsampled datasets.

### Quick Start

```bash
# Run full cycle on Qwen (Normal mode)
python research_engine.py --model Qwen/Qwen2.5-0.5B

# Run rapid validation (Turbo mode - 5 samples)
python research_engine.py --model Qwen/Qwen2.5-0.5B --turbo
```

## 4. Results

Results are saved to `results_g_factor_<model_name>.json`.
See `RESULTS.md` for the latest findings.

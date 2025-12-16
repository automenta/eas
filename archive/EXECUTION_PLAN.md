# Complete Execution Plan: Activation Space Research

> **Purpose:** A step-by-step execution guide requiring zero decision-making. Follow exactly. Every contingency is covered.

---

## Phase 0: Setup (30 minutes)

### Step 0.1: Environment Setup

```bash
cd /home/me/eas
python -m venv .venv
source .venv/bin/activate
pip install torch transformers datasets scikit-learn scipy tqdm numpy matplotlib seaborn
```

**If GPU available:**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Step 0.2: Verify Setup

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
python -c "from transformers import AutoModelForCausalLM; print('Transformers: OK')"
python -c "from datasets import load_dataset; print('Datasets: OK')"
```

**If any fails:** Fix the specific package. Do not proceed until all print "OK" or "True".

### Step 0.3: Create Results Directory

```bash
mkdir -p results/pre_mortems
mkdir -p results/experiments
mkdir -p results/figures
mkdir -p paper
```

---

## Phase 1: Pre-Mortems (2-4 hours)

### Step 1.1: Run Pre-Mortem 1 (Cross-Model Similarity)

```bash
python pre_mortem_1_cross_model.py 2>&1 | tee results/pre_mortems/pm1_output.txt
```

**Record in results/pre_mortems/pm1_summary.md:**
```markdown
# Pre-Mortem 1: Cross-Model Similarity
- Date: [YYYY-MM-DD]
- Duration: [X minutes]
- Average correlation: [X.XXX]
- Result: [PASS/INCONCLUSIVE/FAIL]
```

### Step 1.2: Run Pre-Mortem 2 (Error Prediction)

```bash
python pre_mortem_2_error_prediction.py 2>&1 | tee results/pre_mortems/pm2_output.txt
```

**Record in results/pre_mortems/pm2_summary.md:**
```markdown
# Pre-Mortem 2: Error Prediction
- Date: [YYYY-MM-DD]
- Duration: [X minutes]
- Probe accuracy: [X.XXX]
- Result: [PASS/INCONCLUSIVE/FAIL]
```

### Step 1.3: Run Pre-Mortem 3 (Activation Patching)

```bash
python pre_mortem_3_patching.py 2>&1 | tee results/pre_mortems/pm3_output.txt
```

**Record in results/pre_mortems/pm3_summary.md:**
```markdown
# Pre-Mortem 3: Activation Patching
- Date: [YYYY-MM-DD]
- Duration: [X minutes]
- Change rate: [X.X%]
- Result: [PASS/INCONCLUSIVE/FAIL]
```

### Step 1.4: Decision Point

Count results:

| PM1 | PM2 | PM3 | Action |
|-----|-----|-----|--------|
| FAIL | FAIL | FAIL | **STOP. Go to Phase X (Abort).** |
| FAIL | FAIL | PASS | Go to Phase 2C (Patching only) |
| FAIL | PASS | FAIL | Go to Phase 2B (Error prediction only) |
| FAIL | PASS | PASS | Go to Phase 2BC (Error + Patching) |
| PASS | FAIL | FAIL | Go to Phase 2A (Cross-model only) |
| PASS | FAIL | PASS | Go to Phase 2AC (Cross-model + Patching) |
| PASS | PASS | FAIL | Go to Phase 2AB (Cross-model + Error) |
| PASS | PASS | PASS | **Go to Phase 2ABC (All three).** |

**If any INCONCLUSIVE:** Treat as FAIL for planning, but note for paper discussion.

---

## Phase 2A: Cross-Model Transfer Experiments

**Only proceed here if Pre-Mortem 1 PASSED.**

### Step 2A.1: Extended Cross-Model Analysis

```bash
python experiments/cross_model_extended.py --models gpt2,pythia-70m,pythia-160m,opt-125m --n_samples 2000 2>&1 | tee results/experiments/cross_model_extended.txt
```

**Expected output:** Correlation matrix between all model pairs.

### Step 2A.2: Activation Transfer Experiment

```bash
python experiments/activation_transfer.py --source gpt2 --target pythia-70m --task hellaswag --n_samples 500 2>&1 | tee results/experiments/transfer_results.txt
```

**Success criterion:** Target model accuracy improves by ≥2% with transferred activations.

### Step 2A.3: If Transfer Works

Record metrics:
- Source model baseline accuracy
- Target model baseline accuracy
- Target model + transfer accuracy
- Improvement: [X.X%]

**Go to Phase 3A.**

### Step 2A.4: If Transfer Doesn't Work

Document failure:
- What was tried
- Specific numbers
- Why it likely failed

**Note as negative result. Proceed to other passing directions if any.**

---

## Phase 2B: Error Prediction Experiments

**Only proceed here if Pre-Mortem 2 PASSED.**

### Step 2B.1: Multi-Dataset Error Prediction

```bash
python experiments/error_prediction_extended.py --datasets hellaswag,truthfulqa,mmlu --model gpt2 --n_samples 1000 2>&1 | tee results/experiments/error_pred_extended.txt
```

### Step 2B.2: Unsupervised Discovery (CCS-style)

```bash
python experiments/unsupervised_error.py --model gpt2 --n_samples 500 2>&1 | tee results/experiments/unsupervised_error.txt
```

**Success criterion:** Unsupervised method achieves ≥80% of supervised probe accuracy.

### Step 2B.3: Use for Abstention

```bash
python experiments/abstention_with_probe.py --model gpt2 --dataset hellaswag --abstention_threshold 0.3 2>&1 | tee results/experiments/abstention_results.txt
```

**Success criterion:** Abstention improves effective accuracy (correct / non-abstained).

### Step 2B.4: Document Results

Record:
- Per-dataset probe accuracy
- Unsupervised vs supervised comparison
- Abstention effectiveness

**Go to Phase 3B.**

---

## Phase 2C: Activation Patching Experiments

**Only proceed here if Pre-Mortem 3 PASSED.**

### Step 2C.1: Layer-by-Layer Analysis

```bash
python experiments/patching_layer_analysis.py --model gpt2 --prompts 50 2>&1 | tee results/experiments/patching_layers.txt
```

**Find:** Which layer has strongest causal effect.

### Step 2C.2: Targeted Behavior Editing

```bash
python experiments/patching_behavior_edit.py --model gpt2 --behavior toxicity --n_samples 100 2>&1 | tee results/experiments/patching_toxicity.txt

python experiments/patching_behavior_edit.py --model gpt2 --behavior formality --n_samples 100 2>&1 | tee results/experiments/patching_formality.txt
```

**Success criterion:** Toxicity/formality measurably changes with patching.

### Step 2C.3: Compare to Baselines

```bash
python experiments/patching_vs_prompting.py --model gpt2 --behavior toxicity 2>&1 | tee results/experiments/patching_vs_prompting.txt
```

**Record:** How does patching compare to prompting for the same goal?

**Go to Phase 3C.**

---

## Phase 3: Analysis and Figures

### Step 3.1: Generate Visualizations

```bash
python analysis/generate_figures.py --results_dir results/experiments --output_dir results/figures
```

**Expected outputs:**
- `results/figures/cross_model_heatmap.png`
- `results/figures/error_prediction_roc.png`
- `results/figures/patching_effect_by_layer.png`

### Step 3.2: Compute Statistics

```bash
python analysis/compute_statistics.py --results_dir results/experiments --output results/statistics.json
```

**Record:**
- Effect sizes (Cohen's d)
- p-values
- Confidence intervals

### Step 3.3: Compile Results Table

Create `results/summary_table.md`:

```markdown
| Experiment | Metric | Value | 95% CI | p-value |
|------------|--------|-------|--------|---------|
| Cross-Model Transfer | Correlation | X.XX | [X.XX, X.XX] | X.XXX |
| Error Prediction | AUC | X.XX | [X.XX, X.XX] | X.XXX |
| Activation Patching | Change Rate | X.X% | [X%, X%] | X.XXX |
```

---

## Phase 4: Paper Writing

### Step 4.1: Select Paper Template

Based on results:

| Strongest Result | Target Venue | Template |
|------------------|--------------|----------|
| Cross-Model Transfer | NeurIPS/ICLR | `paper/templates/neurips.tex` |
| Error Prediction | ACL/EMNLP | `paper/templates/acl.tex` |
| Activation Patching | ICLR/ICML | `paper/templates/iclr.tex` |

### Step 4.2: Paper Outline (Fill In)

```
paper/
├── draft.tex
├── figures/
│   ├── main_result.pdf
│   ├── ablation.pdf
│   └── analysis.pdf
├── abstract.txt
└── related_work.bib
```

### Step 4.3: Abstract Template

**If Cross-Model Transfer worked:**
```
We demonstrate that language model activations exhibit cross-model structure,
enabling zero-shot skill transfer between architectures. By learning linear
projections between activation spaces, we transfer [TASK] capability from
[SOURCE] to [TARGET], achieving [X%] improvement over baseline without
fine-tuning. Our findings support the Platonic Representation Hypothesis
and open new avenues for efficient knowledge transfer.
```

**If Error Prediction worked:**
```
We show that language model activations encode predictors of output
correctness. A simple linear probe on intermediate activations achieves
[X%] accuracy at predicting errors, enabling selective abstention that
improves effective accuracy by [X%]. Our unsupervised variant achieves
[X%] of supervised performance without correctness labels.
```

**If Patching worked:**
```
We present activation patching as a method for targeted behavior editing
in language models. By surgically replacing activations at layer [L],
we achieve [X%] change in [BEHAVIOR] without affecting other capabilities.
This provides fine-grained control beyond prompting, with [X%] less
computational cost than fine-tuning.
```

### Step 4.4: Section Checklist

- [ ] Title (working title, refine later)
- [ ] Abstract (200 words max)
- [ ] Introduction (1 page)
  - [ ] Problem statement
  - [ ] Key insight
  - [ ] Contributions (3 bullets)
- [ ] Related Work (0.5-1 page)
  - [ ] Representation similarity
  - [ ] Probing/interpretability
  - [ ] Activation intervention
- [ ] Methods (1-2 pages)
  - [ ] Activation extraction
  - [ ] Main method
  - [ ] Baselines
- [ ] Experiments (2-3 pages)
  - [ ] Setup (models, data)
  - [ ] Main results (table + figure)
  - [ ] Ablations
- [ ] Analysis (0.5-1 page)
  - [ ] What the method learns
  - [ ] Failure cases
- [ ] Conclusion (0.5 page)
- [ ] Appendix
  - [ ] Full hyperparameters
  - [ ] Additional results

---

## Phase X: Abort Procedure

**If all pre-mortems FAIL:**

### Step X.1: Document Negative Results

Create `results/negative_results.md`:

```markdown
# Negative Results: Activation-Based Approaches on Small LMs

## Summary
We tested three activation-based approaches on models ranging from 70M to 345M
parameters. All approaches failed to show meaningful signal.

## Findings

### Cross-Model Similarity
- Correlation after projection: X.XX (threshold: 0.3)
- Interpretation: Activations are model-specific at this scale

### Error Prediction
- Probe accuracy: X.XX (threshold: 0.55)
- Interpretation: Activations don't encode correctness signal

### Activation Patching
- Change rate: X.X% (threshold: 50%)
- Interpretation: Causal structure not localized enough

## Implications
Activation-based approaches may require larger models (7B+) or different
architectures. Future work should test on:
- Models ≥7B parameters
- Encoder-only models (BERT-family)
- Instruction-tuned models

## Value of Negative Results
These findings save other researchers from attempting similar approaches
on small models, and provide baseline expectations for future work.
```

### Step X.2: Consider Pivot

Options:
1. **Scale up:** Try same experiments on larger models (requires more compute)
2. **Different approach:** Prompt-based methods, fine-tuning, distillation
3. **Different task:** Focus on where small models DO work (sentiment, NLI)
4. **Stop:** Accept this line of research is unfruitful

### Step X.3: If Pivoting

Select new direction and return to Phase 0 with new plan.

---

## Contingency Handlers

### C1: Out of Memory Error

```bash
# Reduce batch size
python [script].py --batch_size 8  # or 4, or 1

# Use CPU
python [script].py --device cpu

# Use smaller model
python [script].py --model pythia-70m  # instead of gpt2
```

### C2: Missing Package

```bash
pip install [package_name]
```

### C3: Hugging Face Rate Limit

```bash
# Set token
export HF_TOKEN=your_token_here

# Or use cache
export HF_DATASETS_CACHE=/path/to/large/disk
```

### C4: Results Are Noisy

Run with multiple seeds:
```bash
for seed in 42 123 456 789 1000; do
    python [script].py --seed $seed 2>&1 | tee results/experiments/run_seed_$seed.txt
done
```

Aggregate results and compute mean ± std.

### C5: Results Are Opposite of Expected

Document as negative result. Do NOT cherry-pick. Negative results are valid.

### C6: Crash Mid-Experiment

Experiments should save checkpoints:
```python
# Add to all experiment scripts
import pickle
pickle.dump(results_so_far, open("checkpoint.pkl", "wb"))
```

Resume:
```python
results_so_far = pickle.load(open("checkpoint.pkl", "rb"))
```

### C7: Metric Is Ambiguous

Use multiple metrics. Report all. Let reader judge.

### C8: Reviewer Will Ask "What About X?"

Preemptively test:
- Different models (GPT-2, Pythia, OPT)
- Different layers (first, middle, last)
- Different pooling (mean, last, attention-weighted)
- Different seeds (at least 3)

---

## Timeline

| Day | Activity | Expected Duration |
|-----|----------|-------------------|
| 1 | Phase 0 (Setup) + Phase 1 (Pre-Mortems) | 4-6 hours |
| 2 | Phase 2 (Experiments for passing directions) | 6-8 hours |
| 3 | Phase 2 continued + Phase 3 (Analysis) | 6-8 hours |
| 4 | Phase 4 (Paper outline + draft intro) | 4-6 hours |
| 5 | Phase 4 (Paper methods + results) | 4-6 hours |
| 6 | Phase 4 (Paper polish + submission prep) | 4-6 hours |

**Total:** ~6 days for complete execution

---

## Checklist

### Pre-Mortems
- [ ] PM1 ran, result recorded
- [ ] PM2 ran, result recorded
- [ ] PM3 ran, result recorded
- [ ] Decision made per table

### Experiments (based on passing PMs)
- [ ] Extended analysis completed
- [ ] Main experiment completed
- [ ] Ablations completed
- [ ] Baselines compared

### Analysis
- [ ] Figures generated
- [ ] Statistics computed
- [ ] Summary table created

### Paper
- [ ] Template selected
- [ ] All sections drafted
- [ ] Figures embedded
- [ ] References complete
- [ ] Proofread

---

## Files to Create

Before starting, ensure these files exist (copy from README3.md or create):

```bash
# Pre-mortem scripts
ls pre_mortem_1_cross_model.py  # Must exist before Phase 1
ls pre_mortem_2_error_prediction.py  # Must exist before Phase 1
ls pre_mortem_3_patching.py  # Must exist before Phase 1

# Experiment scripts (create as needed based on passing pre-mortems)
mkdir -p experiments
touch experiments/cross_model_extended.py
touch experiments/activation_transfer.py
touch experiments/error_prediction_extended.py
touch experiments/unsupervised_error.py
touch experiments/abstention_with_probe.py
touch experiments/patching_layer_analysis.py
touch experiments/patching_behavior_edit.py
touch experiments/patching_vs_prompting.py

# Analysis scripts
mkdir -p analysis
touch analysis/generate_figures.py
touch analysis/compute_statistics.py
```

---

**This plan is complete. Follow it exactly. Every step. Every contingency. No improvisation needed.**

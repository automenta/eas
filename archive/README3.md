# Activation Space Archaeology: A Grounded Research Agenda

> **Purpose:** A self-contained research proposal for investigating whether language model activations contain transferable, predictable, and controllable structure. Includes executable code, clear success criteria, and explicit failure points.

---

## 1. Executive Summary

### The Opportunity

Language models encode knowledge and capabilities in their internal activations, but this structure is largely unexplored as a **first-class research artifact**. Three promising directions:

1. **Cross-Model Transfer** — Do models learn similar representations? Can we transfer skills via activations?
2. **Error Prediction** — Can activation geometry predict when the model will be wrong?
3. **Causal Control** — Can we surgically edit behavior by patching activations?

### Prior Failures (What Not To Do)

| Approach | Why It Failed |
|----------|---------------|
| Small residual steering | Too weak to shift token probabilities |
| Style transfer | Sampling dominates; steering invisible |
| Learning from random-chance successes | No signal to learn from |

**Key lesson:** Interventions must be measurable and strong. Pre-mortem tests must validate feasibility before investing time.

---

## 2. Background and Related Work

### Representation Structure in Neural Networks

**Platonic Representation Hypothesis** (Huh et al., 2024): Different neural networks trained on similar tasks converge to similar internal representations. Implications:
- Activations may be more universal than weights
- Cross-model transfer could be possible

**Model Stitching** (Csiszárik et al., 2021): Layers from different models can be combined with learned projections while maintaining reasonable performance. Suggests:
- Activation spaces have compatible geometry
- Linear transforms can bridge model differences

### Probing for Knowledge

**Truthfulness Probes** (Burns et al., 2022): Linear classifiers can detect whether model activations correspond to true or false statements. Implies:
- Activations encode semantic truth
- Simple probes can extract this signal

**Contrast-Consistent Search (CCS)**: Unsupervised method to find truthfulness directions without labels. Shows:
- Activation structure reveals properties beyond supervised learning

### Causal Intervention

**Activation Patching** (Wang et al., 2022): Swapping activations between forward passes identifies causal components. Used for:
- Identifying which layers/positions matter for specific behaviors
- Surgical editing of model behavior

**ROME/MEMIT** (Meng et al., 2022): Editing factual associations by modifying specific activations. Demonstrates:
- Localized intervention can change specific outputs
- Not all activations are equally important

---

## 3. Research Questions

### RQ1: Cross-Model Activation Similarity
> Do different language models develop geometrically similar activation patterns for semantically similar inputs?

**Hypothesis:** If representations converge, we can learn a linear projection that aligns activations across models.

**Testable prediction:** Projected activations from Model A correlate with Model B activations (r > 0.3).

### RQ2: Error Prediction from Activation Geometry
> Can we predict whether a model's output will be correct by examining its activation patterns?

**Hypothesis:** Correct and incorrect predictions occupy different regions of activation space.

**Testable prediction:** A linear classifier on activations achieves >55% accuracy at predicting correctness.

### RQ3: Causal Control via Activation Patching
> Can surgically replacing activations causally change model outputs?

**Hypothesis:** Patching activations from one forward pass into another changes output toward the patched source.

**Testable prediction:** Patching changes output >50% of the time in a controlled setting.

---

## 4. Methods

### 4.1 Models and Data

**Models (diverse architectures, similar scale):**
- GPT-2 (124M parameters)
- Pythia-70M 
- Pythia-160M
- OPT-125M (optional validation)

**Datasets:**
- Wikitext-2 (language modeling)
- HellaSwag (commonsense, with labels)
- TruthfulQA (truthfulness)
- Custom prompts for generation analysis

**Rationale:** Multiple model families test generalization. Mix of language modeling and QA tests different activation usage patterns.

### 4.2 Activation Extraction

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def extract_activation(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    text: str,
    layer: int = -1,
    pooling: str = "mean"
) -> torch.Tensor:
    """
    Extract activation from specified layer.
    
    Args:
        model: HuggingFace causal LM
        tokenizer: Corresponding tokenizer
        text: Input text
        layer: Which layer (-1 = last, -2 = second-to-last, etc.)
        pooling: "mean" (average over sequence), "last" (last token only)
    
    Returns:
        Tensor of shape [hidden_dim]
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    
    hidden = outputs.hidden_states[layer]  # [batch, seq_len, hidden_dim]
    
    if pooling == "mean":
        return hidden.mean(dim=1).squeeze()  # [hidden_dim]
    else:  # "last"
        return hidden[:, -1, :].squeeze()  # [hidden_dim]
```

### 4.3 Pre-Mortem Tests (Detailed)

Each test must pass before proceeding. Estimated time: 2-4 hours each.

---

## 5. Pre-Mortem Test 1: Cross-Model Similarity

### Purpose
Determine whether activations from different models share geometric structure.

### Method
1. Run 1000 prompts through GPT-2 and Pythia-70M
2. Extract final-layer activations from both
3. Learn a linear projection from GPT-2 space → Pythia-70M space
4. Measure how well projected GPT-2 activations predict Pythia-70M activations

### Complete Implementation

```python
#!/usr/bin/env python3
"""
pre_mortem_1_cross_model.py

Tests whether GPT-2 and Pythia-70M have similar activation geometry.
PASS: Average correlation > 0.3 after learned projection
FAIL: Correlation < 0.1
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
from datasets import load_dataset
from tqdm import tqdm


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def extract_activation(model, tokenizer, text, layer=-1):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    
    hidden = outputs.hidden_states[layer]
    return hidden.mean(dim=1).squeeze().cpu().numpy()


def load_prompts(n=1000):
    """Load diverse prompts from Wikitext."""
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    prompts = [t.strip() for t in dataset["text"] if len(t.strip()) > 50]
    return prompts[:n]


def run_pre_mortem_1():
    print("=" * 60)
    print("PRE-MORTEM TEST 1: Cross-Model Activation Similarity")
    print("=" * 60)
    
    device = get_device()
    print(f"Device: {device}")
    
    # Load models
    print("\nLoading GPT-2...")
    gpt2 = AutoModelForCausalLM.from_pretrained("gpt2").to(device).eval()
    gpt2_tok = AutoTokenizer.from_pretrained("gpt2")
    gpt2_tok.pad_token = gpt2_tok.eos_token
    
    print("Loading Pythia-70M...")
    pythia = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-70m").to(device).eval()
    pythia_tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")
    pythia_tok.pad_token = pythia_tok.eos_token
    
    # Load prompts
    prompts = load_prompts(n=500)
    print(f"Loaded {len(prompts)} prompts")
    
    # Extract activations
    print("\nExtracting activations...")
    gpt2_acts = []
    pythia_acts = []
    
    for prompt in tqdm(prompts):
        try:
            g_act = extract_activation(gpt2, gpt2_tok, prompt)
            p_act = extract_activation(pythia, pythia_tok, prompt)
            gpt2_acts.append(g_act)
            pythia_acts.append(p_act)
        except Exception as e:
            continue
    
    X = np.vstack(gpt2_acts)   # [n_samples, gpt2_dim]
    Y = np.vstack(pythia_acts)  # [n_samples, pythia_dim]
    
    print(f"GPT-2 activations: {X.shape}")
    print(f"Pythia activations: {Y.shape}")
    
    # Train/test split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    # Learn projection
    print("\nLearning projection...")
    projector = Ridge(alpha=1.0)
    projector.fit(X_train, Y_train)
    
    # Predict and correlate
    Y_pred = projector.predict(X_test)
    
    # Compute per-dimension correlation
    correlations = []
    for i in range(Y_test.shape[1]):
        r, _ = pearsonr(Y_test[:, i], Y_pred[:, i])
        if not np.isnan(r):
            correlations.append(r)
    
    avg_correlation = np.mean(correlations)
    std_correlation = np.std(correlations)
    
    # Results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Average correlation: {avg_correlation:.3f} (±{std_correlation:.3f})")
    print(f"Min correlation: {np.min(correlations):.3f}")
    print(f"Max correlation: {np.max(correlations):.3f}")
    print(f"Dimensions with r > 0.3: {sum(1 for c in correlations if c > 0.3)}/{len(correlations)}")
    
    # Pass/Fail
    PASS_THRESHOLD = 0.3
    FAIL_THRESHOLD = 0.1
    
    if avg_correlation > PASS_THRESHOLD:
        result = "PASS"
        msg = "Cross-model activation structure exists. Proceed with transfer experiments."
    elif avg_correlation < FAIL_THRESHOLD:
        result = "FAIL"
        msg = "No cross-model structure detected. Abandon this direction."
    else:
        result = "INCONCLUSIVE"
        msg = "Weak signal. Consider more data or different models."
    
    print(f"\n{result}: {msg}")
    print("=" * 60)
    
    return {
        "avg_correlation": avg_correlation,
        "std_correlation": std_correlation,
        "result": result
    }


if __name__ == "__main__":
    run_pre_mortem_1()
```

### Success Criteria

| Metric | Pass | Inconclusive | Fail |
|--------|------|--------------|------|
| Avg correlation | > 0.3 | 0.1 - 0.3 | < 0.1 |

### If Passes
Proceed to activation transfer experiments. Try injecting projected activations from larger models into smaller ones.

### If Fails
Abandon cross-model transfer. Activations are model-specific.

---

## 6. Pre-Mortem Test 2: Error Prediction

### Purpose
Determine whether activation geometry predicts output correctness.

### Complete Implementation

```python
#!/usr/bin/env python3
"""
pre_mortem_2_error_prediction.py

Tests whether activations can predict if the model will be correct.
PASS: Probe accuracy > 55%
FAIL: Probe accuracy ≈ 50% (random chance)
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from datasets import load_dataset
from tqdm import tqdm
import torch.nn.functional as F


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def extract_activation(model, tokenizer, text, layer=-1):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    
    hidden = outputs.hidden_states[layer]
    return hidden.mean(dim=1).squeeze().cpu().numpy()


def get_model_answer(model, tokenizer, prompt, options, device):
    """Get model's MCQ answer by comparing option probabilities."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0, -1, :]
    
    probs = []
    for opt in ["A", "B", "C", "D"]:
        token_ids = tokenizer.encode(f" {opt}", add_special_tokens=False)
        if token_ids:
            prob = F.softmax(logits, dim=-1)[token_ids[0]].item()
        else:
            prob = 0.0
        probs.append(prob)
    
    return probs.index(max(probs))


def load_hellaswag(n=500):
    """Load HellaSwag for testing."""
    dataset = load_dataset("Rowan/hellaswag", split="validation", trust_remote_code=True)
    
    samples = []
    for i, ex in enumerate(dataset):
        if i >= n:
            break
        
        prompt = f"Context: {ex['ctx']}\n"
        prompt += "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(ex['endings'])])
        prompt += "\nAnswer:"
        
        samples.append({
            "prompt": prompt,
            "correct_idx": int(ex["label"]),
            "activation_prompt": ex["ctx"]  # Use context for activation
        })
    
    return samples


def run_pre_mortem_2():
    print("=" * 60)
    print("PRE-MORTEM TEST 2: Error Prediction from Activations")
    print("=" * 60)
    
    device = get_device()
    print(f"Device: {device}")
    
    # Load model
    print("\nLoading GPT-2...")
    model = AutoModelForCausalLM.from_pretrained("gpt2").to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load data
    samples = load_hellaswag(n=500)
    print(f"Loaded {len(samples)} samples")
    
    # Collect activations and correctness labels
    print("\nCollecting activations and predictions...")
    activations = []
    labels = []
    
    for sample in tqdm(samples):
        try:
            # Get activation BEFORE seeing options (represents model's "understanding")
            act = extract_activation(model, tokenizer, sample["activation_prompt"])
            
            # Get model's answer
            pred_idx = get_model_answer(model, tokenizer, sample["prompt"], 4, device)
            is_correct = int(pred_idx == sample["correct_idx"])
            
            activations.append(act)
            labels.append(is_correct)
        except Exception as e:
            continue
    
    X = np.vstack(activations)
    y = np.array(labels)
    
    print(f"Correct predictions: {sum(y)}/{len(y)} ({100*sum(y)/len(y):.1f}%)")
    
    # Train probe with cross-validation
    print("\nTraining error prediction probe...")
    clf = LogisticRegression(max_iter=1000, random_state=42)
    scores = cross_val_score(clf, X, y, cv=5, scoring="accuracy")
    
    avg_accuracy = np.mean(scores)
    std_accuracy = np.std(scores)
    
    # Results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Probe accuracy: {avg_accuracy:.3f} (±{std_accuracy:.3f})")
    print(f"Baseline (random): 0.500")
    print(f"Lift over random: {(avg_accuracy - 0.5) * 100:.1f} percentage points")
    
    # Pass/Fail
    if avg_accuracy > 0.55:
        result = "PASS"
        msg = "Activations encode correctness. Proceed with self-aware uncertainty."
    elif avg_accuracy > 0.52:
        result = "INCONCLUSIVE"
        msg = "Weak signal. May need more data or different layer."
    else:
        result = "FAIL"
        msg = "Activations don't predict correctness. Abandon this direction."
    
    print(f"\n{result}: {msg}")
    print("=" * 60)
    
    return {
        "avg_accuracy": avg_accuracy,
        "std_accuracy": std_accuracy,
        "result": result
    }


if __name__ == "__main__":
    run_pre_mortem_2()
```

### Success Criteria

| Metric | Pass | Inconclusive | Fail |
|--------|------|--------------|------|
| Probe accuracy | > 55% | 52-55% | ≤ 52% |

---

## 7. Pre-Mortem Test 3: Activation Patching

### Purpose
Determine whether patching activations causally affects output.

### Complete Implementation

```python
#!/usr/bin/env python3
"""
pre_mortem_3_patching.py

Tests whether patching activations causally changes output.
PASS: Patching changes output > 50% of the time
FAIL: Patching rarely or never changes output
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional, Dict, Any
from tqdm import tqdm


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


class ActivationPatcher:
    """Hook-based activation patching."""
    
    def __init__(self, model, layer_idx: int):
        self.model = model
        self.layer_idx = layer_idx
        self.stored_activation: Optional[torch.Tensor] = None
        self.patch_activation: Optional[torch.Tensor] = None
        self.mode = "store"  # "store" or "patch"
        self._hook_handle = None
        self._setup_hook()
    
    def _setup_hook(self):
        if hasattr(self.model, 'transformer'):
            layers = self.model.transformer.h
        elif hasattr(self.model, 'gpt_neox'):
            layers = self.model.gpt_neox.layers
        else:
            raise ValueError("Unknown model architecture")
        
        target_layer = layers[self.layer_idx]
        self._hook_handle = target_layer.register_forward_hook(self._hook_fn)
    
    def _hook_fn(self, module, input, output):
        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output
        
        if self.mode == "store":
            self.stored_activation = hidden.detach().clone()
            return output
        elif self.mode == "patch" and self.patch_activation is not None:
            # Replace activation with stored patch
            if isinstance(output, tuple):
                return (self.patch_activation,) + output[1:]
            else:
                return self.patch_activation
        return output
    
    def store_mode(self):
        self.mode = "store"
    
    def patch_mode(self, activation: torch.Tensor):
        self.mode = "patch"
        self.patch_activation = activation
    
    def cleanup(self):
        if self._hook_handle:
            self._hook_handle.remove()


def run_pre_mortem_3():
    print("=" * 60)
    print("PRE-MORTEM TEST 3: Activation Patching Effect")
    print("=" * 60)
    
    device = get_device()
    print(f"Device: {device}")
    
    # Load model
    print("\nLoading GPT-2...")
    model = AutoModelForCausalLM.from_pretrained("gpt2").to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Test prompts
    prompts = [
        "The capital of France is",
        "In 1969, humans first",
        "The largest planet in our solar system is",
        "Water freezes at",
        "The speed of light is approximately",
        "Shakespeare wrote",
        "The mitochondria is the",
        "E equals m c",
        "The first president of the United States was",
        "Photosynthesis occurs in",
    ]
    
    # Test patching at different layers
    n_layers = len(model.transformer.h)
    layer_to_test = n_layers // 2  # Middle layer
    
    print(f"\nTesting patching at layer {layer_to_test}/{n_layers}")
    print("=" * 60)
    
    successful_patches = 0
    total_tests = 0
    
    for prompt in tqdm(prompts):
        try:
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            
            # === Run A: Generate normally, store activation ===
            patcher = ActivationPatcher(model, layer_to_test)
            patcher.store_mode()
            
            with torch.no_grad():
                output_a = model.generate(
                    inputs.input_ids,
                    max_new_tokens=10,
                    do_sample=True,
                    temperature=1.0,
                    pad_token_id=tokenizer.eos_token_id
                )
            text_a = tokenizer.decode(output_a[0], skip_special_tokens=True)
            activation_a = patcher.stored_activation.clone()
            patcher.cleanup()
            
            # === Run B: Generate normally (different output due to sampling) ===
            patcher = ActivationPatcher(model, layer_to_test)
            patcher.store_mode()
            
            with torch.no_grad():
                output_b = model.generate(
                    inputs.input_ids,
                    max_new_tokens=10,
                    do_sample=True,
                    temperature=1.0,
                    pad_token_id=tokenizer.eos_token_id
                )
            text_b = tokenizer.decode(output_b[0], skip_special_tokens=True)
            activation_b = patcher.stored_activation.clone()
            patcher.cleanup()
            
            # === Run C: Patch activation from A into B's generation ===
            patcher = ActivationPatcher(model, layer_to_test)
            patcher.patch_mode(activation_a)
            
            with torch.no_grad():
                output_c = model.generate(
                    inputs.input_ids,
                    max_new_tokens=10,
                    do_sample=False,  # Deterministic to see patch effect
                    pad_token_id=tokenizer.eos_token_id
                )
            text_c = tokenizer.decode(output_c[0], skip_special_tokens=True)
            patcher.cleanup()
            
            # === Check if patching changed output ===
            # Compare C to a fresh deterministic run
            with torch.no_grad():
                output_base = model.generate(
                    inputs.input_ids,
                    max_new_tokens=10,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )
            text_base = tokenizer.decode(output_base[0], skip_special_tokens=True)
            
            changed = (text_c != text_base)
            
            if changed:
                successful_patches += 1
            total_tests += 1
            
            print(f"\n  Prompt: {prompt}")
            print(f"  Base output: {text_base[len(prompt):][:40]}...")
            print(f"  Patched output: {text_c[len(prompt):][:40]}...")
            print(f"  Changed: {changed}")
            
        except Exception as e:
            print(f"  Error: {e}")
            continue
    
    # Results
    change_rate = successful_patches / total_tests if total_tests > 0 else 0
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Patching changed output: {successful_patches}/{total_tests} ({change_rate:.1%})")
    
    if change_rate > 0.5:
        result = "PASS"
        msg = "Patching causally affects output. Proceed with targeted edits."
    elif change_rate > 0.2:
        result = "INCONCLUSIVE"
        msg = "Partial effect. May need different layer or stronger patch."
    else:
        result = "FAIL"
        msg = "Patching doesn't reliably change output. Abandon this direction."
    
    print(f"\n{result}: {msg}")
    print("=" * 60)
    
    return {
        "change_rate": change_rate,
        "successful": successful_patches,
        "total": total_tests,
        "result": result
    }


if __name__ == "__main__":
    run_pre_mortem_3()
```

---

## 8. Decision Framework

After running all three pre-mortems:

```
┌─────────────────────────────────────────────────────────────┐
│  DECISION MATRIX                                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  If 3/3 PASS:                                               │
│    → High-confidence research program                       │
│    → Pursue all three directions in parallel                │
│    → Expect 2+ papers                                       │
│                                                             │
│  If 2/3 PASS:                                               │
│    → Focus on passing directions only                       │
│    → Expect 1-2 papers                                      │
│                                                             │
│  If 1/3 PASS:                                               │
│    → Single-direction focus                                 │
│    → Lower expectations                                     │
│    → Consider pivoting if results are marginal              │
│                                                             │
│  If 0/3 PASS:                                               │
│    → STOP                                                   │
│    → Activation-based approaches unfruitful for small LMs   │
│    → Document negative results                              │
│    → Consider larger models or different approach entirely  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 9. If Pre-Mortems Pass: Next Steps

### For Cross-Model Similarity (if passes)
1. Extend to more model pairs (OPT, LLaMA, Mistral)
2. Test if projected activations improve target model performance
3. Identify which dimensions transfer best (sparse structure?)

### For Error Prediction (if passes)
1. Test on multiple datasets (TruthfulQA, MMLU, ARC)
2. Extend to unsupervised discovery (avoid using correctness labels)
3. Use as uncertainty estimator for abstention

### For Activation Patching (if passes)
1. Identify which layers/positions have strongest causal effects
2. Test for targeted behavior editing (reduce toxicity, increase formality)
3. Compare to prompting and fine-tuning baselines

---

## 10. Setup and Requirements

### Installation

```bash
# Create environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install torch transformers datasets scikit-learn scipy tqdm numpy

# For GPU (recommended)
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Running Pre-Mortems

```bash
# Run all pre-mortems
python pre_mortem_1_cross_model.py
python pre_mortem_2_error_prediction.py
python pre_mortem_3_patching.py
```

### Expected Runtime
- Pre-mortem 1: ~30-60 minutes (500 prompts × 2 models)
- Pre-mortem 2: ~30-60 minutes (500 samples with prediction)
- Pre-mortem 3: ~10-20 minutes (10 prompts × multiple runs)

---

## 11. References

1. **Huh, M., et al. (2024).** The Platonic Representation Hypothesis. *arXiv:2405.07987*
2. **Csiszárik, A., et al. (2021).** Similarity and Matching of Neural Network Representations. *NeurIPS*
3. **Burns, C., et al. (2022).** Discovering Latent Knowledge in Language Models Without Supervision. *arXiv:2212.03827*
4. **Wang, K., et al. (2022).** Interpretability in the Wild: a Circuit for Indirect Object Identification. *arXiv:2211.00593*
5. **Meng, K., et al. (2022).** Locating and Editing Factual Associations in GPT. *NeurIPS*

---

## 12. Success Metrics Summary

| Test | Pass Threshold | Implication |
|------|----------------|-------------|
| Cross-Model Similarity | r > 0.3 | Activation transfer viable |
| Error Prediction | acc > 55% | Self-aware uncertainty possible |
| Activation Patching | change > 50% | Causal control works |

---

## 13. Contact and Contributions

This document is a self-contained research seed. Contributions welcome:
- Running pre-mortems on additional models
- Extending successful directions
- Documenting negative results (equally valuable)

---

*Last updated: 2025-12-13*
*Version: 3.0*

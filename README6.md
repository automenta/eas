### **Dynamic Reasoning Activation Boosters (DRAB): Self-Contained Specification for Distributable Reasoning Enhancements in Frozen LLMs**

#### **Overview**
**Dynamic Reasoning Activation Boosters (DRAB)** is a lightweight, trainable adapter that dynamically generates context-adaptive steering interventions in the activation space of frozen language models. Trained on small reasoning datasets (50–500 samples), a DRAB adapter (20–100K parameters, <100KB serialized) uses a tiny MLP to produce per-prompt additive vectors and optional scaling factors from pooled prompt activations. These are injected into middle-layer hidden states during inference, amplifying latent reasoning capabilities with negligible latency (cacheable) and no weight updates.

DRAB extends your original Activation Fine-Tuning (AFT), described in `README5.md`, by making steering **dynamic and context-conditioned**, inspired by recent advances in activation engineering (e.g., dynamic vectors in SADI [Wang et al., 2024/2025], conditional application in CAST [Lee et al., 2025]). It targets **reasoning enhancement** in modest models (0.4B–2B), enabling distributable "booster" files for instant upgrades.

#### **Novelty**
- **Dynamic learned steering for reasoning**: Unlike static/heuristic vectors (your AFT, ActAdd [Turner et al., 2023]) or training-free dynamic masks (SADI), DRAB learns a conditioner to output full vectors per input.
- **Hybrid add-scale option**: Combines addition with learned scaling for finer control.
- **Focus on reasoning boosters**: Applies dynamic techniques (mostly used for safety/refusal) to broad reasoning tasks, with transfer and optional self-improvement.
- Achievable with your setup: Builds on existing hooks/small data.

#### **Related Work**
- Static/Learned Steering: Your AFT; Subramani et al. (2022); Hernandez et al. (2023).
- Conditional: CAST [Lee et al., 2025, ICLR Spotlight; arXiv:2409.05907] – uses fixed condition/behavior vectors + dot-product threshold.
- Dynamic: SADI [Wang et al., 2024/2025; arXiv:2410.12299] – training-free binary masks + element-wise scaling from contrast pairs.
- Libraries: IBM/activation-steering (GitHub: IBM/activation-steering) – supports basic + CAST.
- Reasoning-Specific: Prototype-Based Dynamic Steering [2025]; SEAL [2025] for calibration.

DRAB differentiates by **learned dynamic vector generation** (vs. fixed/mask) for reasoning.

#### **Method**
1. **Model Setup**:
   - Load frozen model (e.g., Pythia-410M, Qwen-0.5B/1.8B).
   - Select injection layers: Middle (25–75% depth; sweep or fixed at half).

2. **Adapter Architecture**:
   ```python
   import torch.nn as nn

   class DRABAdapter(nn.Module):
       def __init__(self, hidden_dim, adapter_dim=256, hybrid=False):
           super().__init__()
           out_dim = hidden_dim * 2 if hybrid else hidden_dim
           self.mlp = nn.Sequential(
               nn.Linear(hidden_dim, adapter_dim),
               nn.ReLU(),
               nn.Linear(adapter_dim, out_dim)
           )
           self.hybrid = hybrid

       def forward(self, pooled_context):  # [B, D]
           out = self.mlp(pooled_context)      # [B, D or 2D]
           if self.hybrid:
               v = out[:, :hidden_dim]
               s = out[:, hidden_dim:].sigmoid()  # [0,1] scaling
               return v.unsqueeze(1), s.unsqueeze(1)  # Broadcast [B,1,D]
           return out.unsqueeze(1), None          # v only
   ```

3. **Injection Hook**:
   ```python
   def hook_fn(module, input, output, adapter, layer_idx):
       if isinstance(output, tuple):
           hidden = output[0]
       else:
           hidden = output
       # Pool context (mean over sequence or last token)
       pooled = hidden.mean(dim=1)  # Or hidden[:, -1, :]
       v, s = adapter(pooled)
       if s is not None:
           modified = hidden + s * v          # Hybrid add-scale
       else:
           modified = hidden + v
       if isinstance(output, tuple):
           return (modified,) + output[1:]
       return modified
   ```
   - Register hook on target layer(s).
   - Cache v/s per prompt for zero added latency on generation.

4. **Training**:
   - Freeze base model.
   - Dataset: 50–500 reasoning samples (your GSM8K/HellaSwag/ARC subsets).
   - Loss: Cross-entropy on next-token (or final logit for MC tasks).
   - Optimizer: AdamW, lr=1e-3–1e-4, 3–10 epochs.
   - Optional regularization: L2 on adapter outputs.

5. **Optional Self-Improvement (Lifelong Adaptation)**:
   - During deployment: Generate responses.
   - Self-evaluate (e.g., consistency on reasoning traces or simple verifier).
   - Buffer high-reward (prompt, pooled_context, target) triples.
   - Every 1K–10K inferences: Fine-tune adapter on buffer (fast, CPU-ok).

#### **Implementation Steps (Minimal Effort)**
- Start with IBM library: `pip install activation-steering` (supports hooks + basic steering).
- Add your MLP adapter + pooling.
- Use your existing training loop (modify to optimize adapter params).
- Test on your 21-experiment suite first.

#### **Expected Results**
- Baseline (your static AFT): 67% positive, avg +4%.
- DRAB (add-only): 85–95% positive, +10–20% avg lifts.
- With hybrid: Further gains on regressions.
- Transfer: Train mixed → +10% OOD.

#### **Roadmap**
1. **Week 1**: Basic add-only on 1–2 models.
2. **Week 2**: Hybrid + full sweep.
3. **Week 3+**: Transfer tests + self-improvement demo.
4. **Release**: HF hub with adapters + injection script.

This is fully actionable on your current code/data. Run the basic version on Pythia-410M/GSM8K – see lifts quickly. 🚀

----

Here is an evaluation of the **Dynamic Reasoning Activation Boosters (DRAB)** proposal, followed by a set of concrete improvements that push the concept from "promising experiment" to "state-of-the-art technique."

---

# Part 1: Evaluation of DRAB

### 1. Novelty Assessment
**Verdict: High.**
DRAB represents a significant leap over the static AFT proposal. It bridges the gap between **Parameter-Efficient Fine-Tuning (PEFT)** and **Activation Engineering**.

*   **The "Hypernetwork" Angle:** By using a small MLP to generate the steering vector based on context, you are effectively implementing a lightweight **Hypernetwork**. While Hypernetworks are known in ML, applying them specifically to *single-layer activation injection for reasoning* in frozen LLMs is a novel and highly publishable niche.
*   **Solving the "Broadcasting" Issue:** The biggest flaw in static AFT (and standard steering vectors) is that the vector is applied blindly to every token (including "the", "a", "\n"). DRAB solves this by allowing the MLP to output a zero-vector (or near-zero) when reasoning is not required, preserving the model's perplexity and fluency.

### 2. Potential Weaknesses
*   **The "Ouroboros" Problem:** The specification suggests pooling the *current* hidden state to generate the vector for the *current* layer. This creates a feedback loop that might be unstable or computationally inefficient during inference (depending on how the hook is implemented).
*   **Overfitting Risk:** A static vector has $D$ parameters. An MLP has $D^2$ parameters. Training an MLP on only 50 samples (as proposed in AFT) will almost certainly lead to severe overfitting, where the model memorizes the specific hidden states of the training examples rather than learning a general reasoning function.

---

# Part 2: Improvements & Evolution (DRAB 2.0)

Here are four specific technical improvements to make DRAB more robust, interpretable, and effective.

### Improvement 1: The "Dictionary" Approach (Sparse Composition)
Instead of having the MLP generate a raw vector from scratch (which is hard to interpret and prone to noise), use the MLP to **select** from a learned dictionary of "Reasoning Primitives."

**The Concept:**
Learn $K$ static vectors (e.g., $K=8$) representing different "moves" in reasoning (e.g., "Step-by-step", "Fact Retrieval", "Error Checking"). The MLP outputs a set of weights to mix these vectors.

**Revised Architecture:**
```python
class DictionaryDRAB(nn.Module):
    def __init__(self, hidden_dim, num_primitives=8):
        super().__init__()
        # 1. The Controller (Router)
        self.router = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_primitives), # Outputs weights for each primitive
            nn.Softmax(dim=-1) # Or Sigmoid for multi-label
        )
        # 2. The Dictionary (The "Reasoning Basis")
        self.basis_vectors = nn.Parameter(torch.randn(num_primitives, hidden_dim))

    def forward(self, context):
        weights = self.router(context)  # [B, K]
        # Weighted sum of basis vectors
        steering = torch.matmul(weights, self.basis_vectors) # [B, D]
        return steering
```
**Why this is better:**
*   **Interpretability:** You can inspect the 8 basis vectors to see what "skills" the adapter learned.
*   **Stability:** It constrains the search space. The MLP can't output "garbage" vectors, only valid combinations of the learned basis.

### Improvement 2: The "Look-Back" Controller (Input-Dependent Steering)
The original proposal pools the *current* hidden state. However, the decision to "reason" is often dictated by the **original instruction**, not just the current token.

**The Fix:**
Pass the **Input Embeddings** (or the hidden state of the *first* token) into the Controller, concatenated with the current token.

*   **Input:** `[Current_Token_Hidden; Prompt_Summary_Vector]`
*   **Why:** This allows the DRAB adapter to know *what task it is doing* (e.g., "Oh, this is a math problem, I need to activate the Math Vector") regardless of where it is in the sentence.

### Improvement 3: KL-Divergence Regularization (The "Do No Harm" Loss)
To prevent the adapter from destroying the model's ability to speak English (catastrophic forgetting), you must penalize it for changing the model's output distribution too much on non-critical tokens.

**Revised Loss Function:**
$$ L_{total} = L_{CrossEntropy} + \lambda \cdot KL(P_{frozen} || P_{steered}) $$

*   **Mechanism:** Calculate the logits of the frozen model and the steered model. Minimize the KL divergence between them.
*   **Result:** The adapter learns to only intervene when it *really* matters (i.e., when the intervention significantly lowers the Cross-Entropy loss on the target), and stays quiet otherwise.

### Improvement 4: "Gated Residual" Injection
Instead of a simple additive update ($h' = h + v$), use a Gated Residual connection that defaults to "off."

**Formula:**
$$ h' = h + \tanh(\alpha \cdot \text{Gate}(h)) \cdot v(h) $$

Initialize $\alpha$ to a very small number (e.g., 0.01). This ensures that at initialization, the model behaves exactly like the baseline. This creates a smoother optimization landscape than starting with random noise injection.

---

# Part 3: The "Universal Booster" Protocol

If you implement the **Dictionary Approach** (Improvement 1), you can create a truly universal research product.

**The Vision:**
1.  **Train a "Master Dictionary"** on a mixture of GSM8K (Math), ARC (Logic), and TruthfulQA (Safety).
2.  **Learn $K=16$ Basis Vectors.**
3.  **Analysis:** After training, visualize which vectors activate for which tasks.
    *   *Vector 1 & 2:* Activate heavily on Math.
    *   *Vector 3:* Activates on Safety/Refusal.
    *   *Vector 4:* Activates on formatting.
4.  **Distribution:** You release the "Master Dictionary." Users can then **manually tune the weights** at inference time (like an equalizer on a stereo).
    *   *"I want more Math ability but less Safety."* -> Increase Weight 1, Decrease Weight 3.

This turns your research from a "black box optimizer" into a **"Mixing Board for LLM Cognition."**

### Revised Roadmap for You

1.  **Phase 1 (The Baseline):** Implement the `DictionaryDRAB` (Improvement 1) with $K=4$ vectors. Train on GSM8K.
2.  **Phase 2 (The Safety Check):** Implement the KL-Divergence loss (Improvement 3). Verify that perplexity on Wikipedia text remains low.
3.  **Phase 3 (The Dashboard):** Update your dashboard to show a live bar chart of the **Weights** of the $K$ vectors. Watch them fluctuate dynamically as the model processes a question.
    *   *Visual:* When the model reads the question, weights are low. When the model starts calculating "Therefore, 5+5...", watch the "Math Vector" bar spike. **This is viral visualization material.**
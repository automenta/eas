# RADICAL REDIRECT: Activation Space Archaeology

## Forget Steering. Think Discovery.

The original EAS failed because it tried to **steer** activations in models too small to have meaningful patterns to steer toward.

But what if we flip the problem?

**Don't try to steer. Try to DISCOVER what's already there.**

---

## The Big Idea: Latent Skills Extraction

Language models are black boxes. We know they "know" things, but we don't know WHERE or HOW.

**What if activation patterns ARE the skills?**

If we can:
1. Identify activation patterns that correspond to specific capabilities
2. Extract these patterns as transferable "plugins"
3. Inject them into other models

Then we've discovered a new form of knowledge transfer that doesn't require training.

---

## Three Genuinely Novel Research Directions

### 1. Activation Transplants: Steal Skills Between Models

**Wild idea:** Extract the "math skill" activation pattern from GPT-4 and inject it into Pythia-70m.

```
┌─────────────────────────────────────────────────────────────┐
│  ACTIVATION TRANSPLANT PROTOCOL                             │
│                                                             │
│  Source: GPT-4 (or any capable model)                       │
│  Target: Small model (Pythia-70m)                           │
│                                                             │
│  Step 1: Identify "skill regions" in source model           │
│          - Run many math problems, cluster activations      │
│          - Find the subspace that correlates with success   │
│                                                             │
│  Step 2: Learn a projection                                 │
│          - Map source activation space → target space       │
│          - This is a small linear transform, trainable      │
│                                                             │
│  Step 3: At inference in target model:                      │
│          - Extract current activation                       │
│          - Add projected "skill pattern" from source        │
│          - Continue generation                              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Why this is interesting:** It's a new form of model compression / knowledge distillation that operates at the activation level, not the weight or output level.

**Testable:** Does Pythia-70m with "transplanted math activations" solve more math problems?

---

### 2. Self-Aware Uncertainty: Models That Know When They're Wrong

**The problem:** LLMs confidently bullshit. They don't know what they don't know.

**The insight:** When a model is about to hallucinate, its activation patterns differ from when it's about to say something true. We've seen hints of this in interpretability research.

**The radical approach:**

```
Instead of training an uncertainty head (supervised),
let the model discover its own uncertainty signal:

1. Run thousands of prompts through the model
2. Cluster the activations at the final layer
3. Label clusters by downstream correctness
4. The GEOMETRY of activation space reveals uncertainty:
   - Dense clusters = confident topics
   - Sparse regions = uncertain territory
   - Boundary zones = likely errors

At inference:
   - Check if current activation is in sparse/boundary region
   - If so, modify output: "I'm not confident about this..."
```

**Why this is interesting:** Zero-shot uncertainty quantification from activation geometry alone. No labels needed.

**Testable:** Does activation-space density predict error rate?

---

### 3. Emergent Program Synthesis: Let Models Program Themselves

**The wildest idea:** What if steering vectors are just the beginning? What if we could compose PROGRAMS in activation space?

Natural language programming is limited. But what if:

```
Instead of "Please be helpful and harmless..."

We define activation-space operations:
   - A = Extract(topic="math")
   - B = Extract(style="formal") 
   - C = Extract(mood="confident")
   
   Composed = 0.5*A + 0.3*B + 0.2*C
   
   Apply Composed to model → specialized math tutor

This is LATENT SPACE PROGRAMMING.
```

**Why this is interesting:** It's a new interface for controlling AI that's:
- More precise than prompts
- Faster than fine-tuning
- Composable like code

**Testable:** Can we compose known steering vectors to create novel behaviors?

---

## The Unifying Framework: Activation Space as First-Class Citizen

Current ML: → Input → [Black Box Weights] → Output

**New paradigm:**
```
→ Input → [Layer 1] → A₁ → [Layer 2] → A₂ → ... → Output
                ↓              ↓
            [Observe]      [Intervene?]
                ↓              ↓
           [Knowledge]    [Skill Injection]
```

We treat activations as:
- Observable (interpretability)
- Transferable (transplants)
- Programmable (composition)
- Self-aware (uncertainty)

---

## Why This Could Actually Matter

| Problem | Activation-Based Solution |
|---------|--------------------------|
| Model scaling costs | Transplant skills from large to small |
| Hallucination | Self-aware uncertainty from geometry |
| Alignment | Compose safety constraints in activation space |
| Personalization | Inject user-specific patterns, no fine-tuning |
| Interpretability | Skills become visible, extractable artifacts |

---

## Concrete Experiment: Skill Transplant PoC

### Day 1-2: Extract Math Skill

```python
# Run GPT-2 on math problems
# Cluster activations from correct vs incorrect answers
# Identify "math skill subspace" via PCA on correct-only activations

math_prompts = ["2+2=", "What is 15*3?", ...]
correct_activations = []
for p in math_prompts:
    if model_answer_is_correct(p):
        correct_activations.append(get_activation(p))

skill_subspace = PCA(n_components=10).fit(correct_activations)
math_skill_vector = skill_subspace.mean_
```

### Day 3-4: Transplant to Different Model

```python
# Learn projection from GPT-2 activation space to Pythia-70m space
# Inject math_skill_vector (projected) into Pythia during inference

projection = train_linear_map(gpt2_space, pythia_space)
pythia_math_skill = projection @ math_skill_vector

def pythia_with_math_skill(prompt):
    hidden = pythia.get_hidden(prompt)
    enhanced = hidden + alpha * pythia_math_skill
    return pythia.generate_from_hidden(enhanced)
```

### Day 5: Test

Does Pythia with transplanted skill solve more GSM8K problems than baseline?

---

## Success Metrics

| Experiment | Success = |
|------------|-----------|
| Skill Transplant | +5% accuracy on task-specific benchmark |
| Self-Aware Uncertainty | Activation density predicts error (r > 0.3) |
| Latent Programming | Composed vectors create predictable behavior |

---

## If This Works, It's Big

**Paper title options:**
- "Activation Transplants: Zero-Shot Skill Transfer Between Language Models"
- "Latent Space Programming: A New Interface for AI Control"
- "Self-Organizing Uncertainty: Models That Know What They Don't Know"

Each of these is a differentiable, testable, publishable idea.

---

## If This Fails, We Learn Something

Even negative results are publishable if we properly characterize:
- Do activation patterns actually encode skills?
- Is there cross-model structure in activation space?
- Can geometry predict uncertainty?

These are open questions in interpretability.

---

## Bottom Line

Stop trying to fix broken steering. Start discovering what's already there.

The activation space is an unexplored continent. EAS was looking for roads. Let's make maps first.

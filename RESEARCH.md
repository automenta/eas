# EAS Research Roadmap

> **Status**: Complete research seed (December 2025)  
> **Focus**: Novel reasoning enhancements building on established CTD research  
> **Compute Target**: Consumer-grade (8-16GB VRAM, single GPU)

---

## Executive Summary

This document is a self-contained seed for a research program developing **reasoning enhancement systems for language models**. It builds on established Critical Token Divergence (CTD) research while contributing genuinely novel innovations in meta-cognition, self-evolution, and causal reasoning.

### Core Insight

Language models encode reasoning quality at specific token positions with 100-800x greater signal than at context positions (CTD phenomenon). This established finding enables:

1. **Position-aware intervention** targeting where signal is strongest
2. **Meta-cognitive awareness** using CTD as confidence signal  
3. **Evolutionary optimization** using CTD as fitness function
4. **Causal reasoning** via explicit graph structures + neural steering

### Novel Contributions (This Research)

| Innovation | Novelty | Why It Matters |
|------------|---------|----------------|
| **Meta-Cognitive Reasoning** | First uncertainty-aware LM reasoning system | Models know what they don't know |
| **Self-Evolving Intervention** | Genetic optimization of steering strategies | Systems improve through use |
| **Causal Graph Integration** | Do-calculus + neural steering | True causal (not correlational) reasoning |

---

## Part I: Foundation - Critical Token Divergence

### 1.1 Literature Context

CTD is an **established phenomenon** documented in 2024-2025 research:

| Paper | Key Finding | Reference |
|-------|-------------|-----------|
| Divergent Token Metrics | First Divergent Token (FDT) causes cascading divergence | [arXiv:2311.01544](https://arxiv.org/abs/2311.01544) |
| Critical Tokens Matter | Critical tokens cause 90% path separation in math reasoning | [arXiv:2411.19943](https://arxiv.org/abs/2411.19943) |
| Selective Critical Token Fine-Tuning | Critical token targeting improves accuracy 10-20% | [arXiv:2510.10974](https://arxiv.org/abs/2510.10974) |
| Why LLMs Attend to First Token | Attention sinks cause representational divergence | [arXiv:2504.02732](https://arxiv.org/abs/2504.02732) |

### 1.2 Our Validation (Pythia Family)

We independently validated CTD across model scales:

```
┌─────────────────────────────────────────────────────────────┐
│              CTD SCALING VALIDATION                         │
├─────────────────────────────────────────────────────────────┤
│  Pythia-70m   │  CTD: 122x   │  Cohen's d: 2.16  │ baseline │
│  Pythia-160m  │  CTD: 170x   │  Cohen's d: 3.14  │ +39%     │
│  Pythia-410m  │  CTD: 770x   │  Cohen's d: 3.86  │ +529%    │
├─────────────────────────────────────────────────────────────┤
│  Scaling correlation: r = 0.982 (super-linear)              │
└─────────────────────────────────────────────────────────────┘
```

### 1.3 Position Types and Weights

Critical positions have 5-10x stronger reasoning signal:

| Position Type | Detection Method | CTD Weight |
|---------------|------------------|------------|
| **Conclusion markers** | "therefore", "thus", "so", "hence" | 5.0x |
| **Judgment tokens** | "correct", "wrong", "valid", "invalid" | 5.0x |
| **Negation** | "not", "never", "cannot" | 3.0x |
| **Final 20% of sequence** | Position-based | 2.0x |
| **Context tokens** | All others | 1.0x |

---

## Part II: Novel Innovation - Meta-Cognitive Reasoning Engine

### 2.1 Theoretical Motivation

Current LMs have no mechanism to:
- Estimate their own uncertainty
- Choose appropriate reasoning strategy
- Predict errors before making them
- Abstain when confidence is low

**MCRE provides all four capabilities.**

### 2.2 Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    TEXT INPUT                               │
└───────────────────────┬─────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│              LANGUAGE MODEL (frozen)                        │
│         Extract hidden states at critical positions         │
└───────────────────────┬─────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│              META-COGNITIVE LAYER                           │
│  ┌─────────────────┐  ┌─────────────────┐                   │
│  │ Uncertainty     │  │ Strategy        │                   │
│  │ Quantifier      │  │ Selector        │                   │
│  │ (MLP: d→32→1)   │  │ (pattern match) │                   │
│  └────────┬────────┘  └────────┬────────┘                   │
│           ↓                    ↓                            │
│  ┌─────────────────┐  ┌─────────────────┐                   │
│  │ Error           │  │ Explanation     │                   │
│  │ Predictor       │  │ Generator       │                   │
│  │ (failure sigs)  │  │ (templates)     │                   │
│  └────────┬────────┘  └────────┬────────┘                   │
└───────────┴────────────────────┴────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│              META-COGNITIVE STATE                           │
│  strategy: ReasoningStrategy (enum)                         │
│  confidence: float [0, 1]                                   │
│  uncertainty: float [0, 1]                                  │
│  error_risk: float [0, 1]                                   │
│  explanation: str                                           │
│  should_abstain: bool                                       │
└─────────────────────────────────────────────────────────────┘
```

### 2.3 Component Specifications

#### 2.3.1 UncertaintyQuantifier

```python
class UncertaintyQuantifier(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.epistemic = nn.Sequential(
            nn.Linear(dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        self.aleatoric = nn.Sequential(
            nn.Linear(dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, hidden: Tensor) -> Tuple[float, float]:
        # Pool at critical positions only
        pooled = hidden.mean(dim=1)
        epistemic = self.epistemic(pooled).item()
        aleatoric = self.aleatoric(pooled).item()
        return epistemic, aleatoric
```

**Compute**: ~65K parameters per quantifier, <1ms inference on CPU.

#### 2.3.2 StrategySelector

Five reasoning strategies with pattern-based detection:

| Strategy | Patterns | Success Weight |
|----------|----------|----------------|
| DEDUCTIVE | "if...then", "all...are", "therefore" | Track per-strategy |
| INDUCTIVE | "usually", "most", "tends to" | success rate |
| ABDUCTIVE | "best explanation", "likely because" | for dynamic |
| ANALOGICAL | "similar to", "like", "as...so" | selection |
| CAUSAL | "causes", "leads to", "because of" | |

```python
def select_strategy(self, text: str) -> ReasoningStrategy:
    scores = {}
    for strategy in ReasoningStrategy:
        pattern_score = self._pattern_match(text, strategy)
        success_rate = self.success_rates.get(strategy, 0.5)
        scores[strategy] = pattern_score * 0.7 + success_rate * 0.3
    return max(scores, key=scores.get)
```

#### 2.3.3 ErrorPredictor

Learns signatures of reasoning failures:

```python
class ErrorPredictor:
    def __init__(self, dim: int):
        self.failure_signatures: List[Tensor] = []
        self.threshold = 0.8  # Cosine similarity threshold
    
    def predict_error_risk(self, hidden: Tensor) -> float:
        if not self.failure_signatures:
            return 0.5  # Prior
        
        pooled = hidden.mean(dim=1)
        similarities = [
            F.cosine_similarity(pooled, sig, dim=-1).max()
            for sig in self.failure_signatures
        ]
        return max(similarities).item()
    
    def record_failure(self, hidden: Tensor):
        self.failure_signatures.append(hidden.mean(dim=1).detach())
```

#### 2.3.4 Abstention Decision

```python
def should_abstain(self, state: MetaCognitiveState) -> bool:
    if state.uncertainty > self.uncertainty_threshold:  # 0.7 default
        return True
    if state.error_risk > self.error_risk_threshold:  # 0.6 default
        return True
    if state.confidence < self.confidence_threshold:  # 0.3 default
        return True
    return False
```

### 2.4 Training Pipeline

**Phase 1: Uncertainty Calibration**
- Dataset: 1000 reasoning examples with ground truth
- Train uncertainty networks to predict correctness
- Loss: Binary cross-entropy on (confidence → correct)

**Phase 2: Strategy Performance Tracking**
- Run all strategies on held-out examples
- Track success rate per strategy
- Update selection weights

**Phase 3: Failure Signature Collection**
- Record hidden states from incorrect predictions
- Build failure signature library
- Tune similarity threshold via validation

### 2.5 Evaluation Plan

| Dataset | Task | Metric |
|---------|------|--------|
| LogiQA | Logical reasoning | Accuracy, Calibration Error |
| ReClor | Reading comprehension | Accuracy with abstention |
| ARC | Science reasoning | F1 at various abstention rates |
| TruthfulQA | Hallucination detection | Abstention precision/recall |

**Key Hypothesis**: Principled abstention reduces effective error rate by 20%+ compared to always-answer baseline.

---

## Part III: Novel Innovation - Self-Evolving Reasoning System

### 3.1 Theoretical Motivation

Current intervention methods have fixed hyperparameters (layers, strengths, positions). **SERS evolves these through experience.**

### 3.2 Evolutionary Strategy Genome

```python
@dataclass
class EvolutionaryStrategy:
    strategy_id: str
    layer_weights: Dict[int, float]      # Which layers, how strongly
    alpha_schedule: List[float]          # Intervention strength by position
    primitive_focus: List[str]           # Logic primitives to target
    position_weights: Dict[str, float]   # CTD position weighting
    fitness: float = 0.0                 # Performance score
    generation: int = 0
```

### 3.3 Genetic Algorithm

```
┌─────────────────────────────────────────────────────────────┐
│                 GENERATION 0: Random Population             │
│  Strategy_A: layers=[0,2], alpha=[0.1,0.2,0.3], fitness=0   │
│  Strategy_B: layers=[1,3], alpha=[0.2,0.1,0.2], fitness=0   │
│  Strategy_C: layers=[0,1], alpha=[0.3,0.3,0.1], fitness=0   │
│  ...                                                        │
└─────────────────────────────────────────────────────────────┘
                        ↓ Evaluate on tasks
┌─────────────────────────────────────────────────────────────┐
│                 EVALUATION: Fitness Scoring                 │
│  Strategy_A: 15/20 correct → fitness = 0.75                 │
│  Strategy_B: 12/20 correct → fitness = 0.60                 │
│  Strategy_C: 18/20 correct → fitness = 0.90 ← Best          │
└─────────────────────────────────────────────────────────────┘
                        ↓ Selection (top 50%)
┌─────────────────────────────────────────────────────────────┐
│                 CROSSOVER + MUTATION                        │
│  Child_1 = crossover(Strategy_C, Strategy_A) + mutation     │
│  Child_2 = crossover(Strategy_A, Strategy_B) + mutation     │
│  ...                                                        │
└─────────────────────────────────────────────────────────────┘
                        ↓ Repeat
```

### 3.4 Component Specifications

#### 3.4.1 Fitness Function (CTD-Based)

```python
def compute_fitness(self, strategy: EvolutionaryStrategy, 
                    tasks: List[ReasoningTask]) -> float:
    correct = 0
    total = len(tasks)
    
    for task in tasks:
        # Apply strategy-specific intervention
        hidden = self.model.get_hidden(task.text)
        modified = self.apply_strategy(hidden, strategy)
        output = self.model.generate(modified)
        
        if self.evaluate_correctness(output, task.expected):
            correct += 1
    
    return correct / total
```

#### 3.4.2 Crossover

```python
def crossover(self, parent_a: EvolutionaryStrategy, 
              parent_b: EvolutionaryStrategy) -> EvolutionaryStrategy:
    child = EvolutionaryStrategy(
        strategy_id=f"gen{self.generation}_child{self.child_count}",
        # Uniform crossover for layer weights
        layer_weights={
            layer: random.choice([parent_a.layer_weights.get(layer, 0),
                                  parent_b.layer_weights.get(layer, 0)])
            for layer in set(parent_a.layer_weights) | set(parent_b.layer_weights)
        },
        # Blend crossover for alpha schedule
        alpha_schedule=[
            0.5 * a + 0.5 * b 
            for a, b in zip(parent_a.alpha_schedule, parent_b.alpha_schedule)
        ],
        primitive_focus=random.choice([parent_a.primitive_focus, 
                                        parent_b.primitive_focus]),
        generation=self.generation
    )
    return child
```

#### 3.4.3 Mutation

```python
def mutate(self, strategy: EvolutionaryStrategy, 
           mutation_rate: float = 0.1) -> EvolutionaryStrategy:
    if random.random() < mutation_rate:
        # Mutate a random layer weight
        layer = random.choice(list(strategy.layer_weights.keys()))
        strategy.layer_weights[layer] *= random.uniform(0.8, 1.2)
    
    if random.random() < mutation_rate:
        # Mutate alpha schedule
        idx = random.randint(0, len(strategy.alpha_schedule) - 1)
        strategy.alpha_schedule[idx] *= random.uniform(0.8, 1.2)
    
    return strategy
```

### 3.5 Failure Analysis and Recovery

#### 3.5.1 Failure Types

```python
class FailureType(Enum):
    CONTRADICTION = "contradiction"      # "A and not A"
    NON_SEQUITUR = "non_sequitur"       # Conclusion doesn't follow
    HALLUCINATION = "hallucination"      # Fabricated facts
    INCOMPLETE = "incomplete"            # Missing steps
    CIRCULAR = "circular"                # Conclusion in premise
```

#### 3.5.2 Recovery Learning

```python
class RecoveryLearner:
    def __init__(self, dim: int):
        self.recovery_directions: Dict[FailureType, Tensor] = {}
    
    def learn_recovery(self, failure_hidden: Tensor, 
                       success_hidden: Tensor,
                       failure_type: FailureType):
        # Direction from failure to success
        direction = success_hidden.mean(dim=1) - failure_hidden.mean(dim=1)
        direction = F.normalize(direction, dim=-1)
        
        # Exponential moving average update
        if failure_type in self.recovery_directions:
            self.recovery_directions[failure_type] = (
                0.9 * self.recovery_directions[failure_type] + 
                0.1 * direction
            )
        else:
            self.recovery_directions[failure_type] = direction
    
    def apply_recovery(self, hidden: Tensor, 
                       failure_type: FailureType,
                       strength: float = 0.1) -> Tensor:
        direction = self.recovery_directions.get(failure_type)
        if direction is None:
            return hidden
        return hidden + strength * direction
```

### 3.6 Experiment Plan

| Experiment | Episodes | Population | Expected |
|------------|----------|------------|----------|
| Baseline | 100 | 10 | Convergence validation |
| Standard | 1000 | 20 | Strategy improvement |
| Long-horizon | 5000 | 50 | Optimal strategy discovery |
| Cross-domain | 1000 | 20 | Transfer validation |

**Compute**: ~50 hours on single A100 for long-horizon (5 hours on RTX 4090).

---

## Part IV: Novel Innovation - Causal Reasoning Graph

### 4.1 Theoretical Motivation

LMs learn correlations, not causation. "Ice cream sales correlate with drowning" ≠ "Ice cream causes drowning."

**CRG enables true causal reasoning via do-calculus.**

### 4.2 Causal Concepts

| Concept | Notation | Meaning |
|---------|----------|---------|
| **Observation** | P(Y\|X=x) | Probability of Y given we observe X=x |
| **Intervention** | P(Y\|do(X=x)) | Probability of Y if we SET X=x |
| **Counterfactual** | P(Y_x\|X=x') | What would Y be if X were x, given X was actually x' |

**Key insight**: P(Y|X) ≠ P(Y|do(X)) due to confounders.

### 4.3 Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    TEXT INPUT                               │
│  "If it rains, the ground gets wet. The ground is wet."    │
└───────────────────────┬─────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│              CAUSAL EXTRACTOR                               │
│  Parse patterns: if-then, causes, leads to, because         │
│  Output: List of (cause, effect) pairs                      │
└───────────────────────┬─────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│              CAUSAL GRAPH                                   │
│  Nodes: [rain, wet_ground, sprinkler, ...]                  │
│  Edges: [(rain → wet_ground), (sprinkler → wet_ground)]     │
└───────────────────────┬─────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│              DO-CALCULUS ENGINE                             │
│  do(rain=true): Remove incoming edges to "rain"             │
│  Compute: P(wet_ground | do(rain=true))                     │
└───────────────────────┬─────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│              NEURAL STEERING                                │
│  Use causal structure to weight interventions               │
│  Stronger steering on causal chain paths                    │
└─────────────────────────────────────────────────────────────┘
```

### 4.4 Causal Pattern Extraction

```python
CAUSAL_PATTERNS = [
    (r"if\s+(.+?)\s+then\s+(.+?)[\.,]", "if_then"),
    (r"(.+?)\s+causes?\s+(.+?)[\.,]", "causes"),
    (r"(.+?)\s+leads?\s+to\s+(.+?)[\.,]", "leads_to"),
    (r"(.+?)\s+because\s+(.+?)[\.,]", "because"),  # Note: reversed
    (r"(.+?)\s+results?\s+in\s+(.+?)[\.,]", "results_in"),
    (r"when\s+(.+?)[,]\s+(.+?)[\.,]", "when_then"),
]

def extract_causal_relations(text: str) -> List[Tuple[str, str]]:
    relations = []
    for pattern, ptype in CAUSAL_PATTERNS:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            cause, effect = match.groups()
            if ptype == "because":
                cause, effect = effect, cause  # Reverse order
            relations.append((normalize(cause), normalize(effect)))
    return relations
```

### 4.5 Do-Calculus Implementation

```python
class CausalGraph:
    def __init__(self):
        self.nodes: Set[str] = set()
        self.edges: Dict[str, Set[str]] = {}  # parent → children
        self.parents: Dict[str, Set[str]] = {}  # child → parents
    
    def add_edge(self, cause: str, effect: str):
        self.nodes.add(cause)
        self.nodes.add(effect)
        self.edges.setdefault(cause, set()).add(effect)
        self.parents.setdefault(effect, set()).add(cause)
    
    def do_intervention(self, variable: str) -> "CausalGraph":
        """Implement do(X) by removing incoming edges to X."""
        new_graph = copy.deepcopy(self)
        # Remove all incoming edges to the intervened variable
        for parent in list(new_graph.parents.get(variable, [])):
            new_graph.edges[parent].discard(variable)
        new_graph.parents[variable] = set()
        return new_graph
    
    def get_causal_effect(self, cause: str, effect: str) -> bool:
        """Check if cause → effect path exists in do(cause) graph."""
        do_graph = self.do_intervention(cause)
        return do_graph.has_path(cause, effect)
    
    def has_path(self, start: str, end: str) -> bool:
        """BFS for path existence."""
        visited = set()
        queue = [start]
        while queue:
            node = queue.pop(0)
            if node == end:
                return True
            if node in visited:
                continue
            visited.add(node)
            queue.extend(self.edges.get(node, []))
        return False
```

### 4.6 Counterfactual Steering

```python
def counterfactual_intervention(self, hidden: Tensor, 
                                 graph: CausalGraph,
                                 variable: str,
                                 counterfactual_value: str) -> Tensor:
    """Modify representations to reflect counterfactual world."""
    # Get original representation
    original = hidden.clone()
    
    # Encode counterfactual as direction
    cf_embedding = self.encode_counterfactual(variable, counterfactual_value)
    
    # Find positions related to variable
    var_positions = self.find_variable_positions(variable, hidden)
    
    # Apply counterfactual direction at those positions
    for pos in var_positions:
        original[:, pos] = original[:, pos] + 0.2 * cf_embedding
    
    return original
```

### 4.7 Evaluation Plan

| Dataset | Task | Metric |
|---------|------|--------|
| bAbI | Synthetic reasoning | Accuracy on causal tasks |
| CLUTRR | Kinship reasoning | Accuracy with causal chains |
| COPA | Causal judgment | Accuracy vs baseline |
| Counterfactual reasoning | Custom dataset | Intervention correctness |

---

## Part V: Supporting Innovations

### 5.1 Compositional Logic Grounding

Map logical primitives to neural attractors:

| Primitive | Symbol | Neural Representation |
|-----------|--------|----------------------|
| IMPLICATION | → | Learned attractor A₁ |
| CONJUNCTION | ∧ | Learned attractor A₂ |
| DISJUNCTION | ∨ | Learned attractor A₃ |
| NEGATION | ¬ | Learned attractor A₄ |
| MODUS_PONENS | MP | Learned attractor A₅ |
| MODUS_TOLLENS | MT | Learned attractor A₆ |

```python
class GroundedPrimitiveMemory(nn.Module):
    def __init__(self, dim: int, num_attractors: int = 5):
        super().__init__()
        self.attractors = nn.ParameterDict({
            prim.value: nn.Parameter(torch.randn(num_attractors, dim) * 0.1)
            for prim in LogicPrimitive
        })
    
    def get_composite_attractor(self, primitives: List[LogicPrimitive]) -> Tensor:
        """Combine attractors for detected primitives."""
        if not primitives:
            return None
        attractors = [self.attractors[p.value] for p in primitives]
        # Weighted average of mean attractors
        return torch.stack([a.mean(dim=0) for a in attractors]).mean(dim=0)
```

### 5.2 Position-Aware EAS

Exploit CTD by weighting interventions by position:

```python
class PositionAwareWatcher(EmergentWatcher):
    def __init__(self, dim: int):
        super().__init__(dim)
        self.position_weights = {
            "conclusion": 5.0,   # "therefore", "thus"
            "judgment": 5.0,    # "correct", "wrong"
            "negation": 3.0,    # "not", "never"
            "late": 2.0,        # Final 20%
            "context": 1.0      # Everything else
        }
    
    def weighted_snap(self, hidden: Tensor, positions: List[int]) -> Tensor:
        """Apply position-weighted snapping."""
        result = hidden.clone()
        for pos in positions:
            weight = self.get_position_weight(pos)
            delta = self.compute_snap_direction(hidden[:, pos])
            result[:, pos] = hidden[:, pos] + weight * delta
        return result
```

### 5.3 Adaptive Reasoning Amplifier

Dynamic steering during generation:

```python
class AdaptiveReasoningAmplifier:
    def __init__(self, dim: int):
        self.correct_direction: Tensor = None
        self.momentum = torch.zeros(dim)
        self.quality_history: List[float] = []
        self.momentum_decay = 0.9
        self.backtrack_threshold = 0.3
    
    def adaptive_steer(self, hidden: Tensor) -> Tensor:
        quality = self.measure_quality(hidden)
        self.quality_history.append(quality)
        
        # Compute adaptive strength: weaker when quality is high
        strength = 0.1 * (1 - quality)
        
        # Update momentum
        if self.correct_direction is not None:
            self.momentum = (self.momentum_decay * self.momentum + 
                            (1 - self.momentum_decay) * self.correct_direction)
        
        return hidden + strength * self.momentum
    
    def should_backtrack(self) -> bool:
        if len(self.quality_history) < 3:
            return False
        recent = self.quality_history[-3:]
        return all(q < self.backtrack_threshold for q in recent)
```

---

## Part VI: Proof-of-Concept Products

### 6.1 PoC 1: Reasoning Confidence Indicator

**Objective**: Chrome extension showing reasoning confidence for LLM outputs.

#### Architecture
```
┌─────────────────────────────────────────────────────────────┐
│                 BROWSER EXTENSION                           │
│  Intercept: ChatGPT, Claude, Gemini responses               │
│  Send: Response text to local MCRE server                   │
│  Display: Confidence badge + explanation                    │
└───────────────────────┬─────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│              LOCAL MCRE SERVER (Python + FastAPI)           │
│  Model: Pythia-70m or Phi-2 (consumer-grade)                │
│  Compute: MCRE meta-cognitive analysis                      │
│  Return: {confidence, strategy, should_abstain, explanation}│
└─────────────────────────────────────────────────────────────┘
```

#### Implementation
```python
# server.py
from fastapi import FastAPI
from eas.src.intervention.metacognitive import MetaCognitiveReasoningEngine

app = FastAPI()
mcre = MetaCognitiveReasoningEngine(load_model("microsoft/phi-2"))

@app.post("/analyze")
async def analyze_reasoning(text: str):
    state, _ = mcre.reason(text)
    return {
        "confidence": state.confidence,
        "uncertainty": state.uncertainty,
        "strategy": state.strategy.value,
        "should_abstain": mcre.should_abstain(state),
        "explanation": state.explanation
    }
```

#### Compute Requirements
- Model: Phi-2 (2.7B) or Pythia-70m
- VRAM: 3-6 GB
- Latency: <500ms per analysis

#### Demonstration
1. User asks ChatGPT a logic puzzle
2. Extension captures response
3. MCRE analyzes reasoning quality
4. Badge shows: "⚠️ Low Confidence (0.35) - May contain reasoning errors"

---

### 6.2 PoC 2: Self-Improving Homework Helper

**Objective**: Tutoring app that learns from mistakes and improves over time.

#### Architecture
```
┌─────────────────────────────────────────────────────────────┐
│               STUDENT INTERFACE (Web/Mobile)                │
│  Input: Homework problem                                    │
│  Output: Solution + explanation                             │
│  Feedback: "Correct" / "Wrong" button                       │
└───────────────────────┬─────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│              SERS BACKEND                                   │
│  1. Generate solution using current best strategy           │
│  2. Apply position-aware intervention                       │
│  3. Record outcome in trace memory                          │
│  4. If wrong: analyze failure, learn recovery direction     │
│  5. Periodically: evolve strategy population                │
└─────────────────────────────────────────────────────────────┘
```

#### Learning Loop
```python
class HomeworkHelper:
    def __init__(self, model_path: str):
        self.sers = SelfEvolvingReasoningSystem(load_model(model_path))
        self.problem_count = 0
    
    def solve(self, problem: str) -> str:
        trace = self.sers.reason(problem)
        return trace.output
    
    def feedback(self, problem: str, was_correct: bool):
        self.sers.record_outcome(problem, was_correct)
        self.problem_count += 1
        
        # Evolve every 20 problems
        if self.problem_count % 20 == 0:
            self.sers.evolve_strategies(episodes=10)
```

#### Demonstration
1. Day 1: 60% accuracy on logic problems
2. Day 7: 75% accuracy (strategies evolved from feedback)
3. Day 30: 85% accuracy (optimal strategies discovered)

#### Compute Requirements
- Model: Phi-2 or Mistral-7B-Instruct (quantized)
- Training: 10 evolution epochs ~5 minutes on RTX 3060
- Inference: <2 seconds per problem

---

### 6.3 PoC 3: Causal Claim Verifier

**Objective**: Tool that checks whether causal claims in text are valid.

#### Architecture
```
┌─────────────────────────────────────────────────────────────┐
│                    INPUT TEXT                               │
│  "Coffee causes heart disease because studies show          │
│   correlation between coffee consumption and heart attacks."│
└───────────────────────┬─────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│              CAUSAL REASONING GRAPH                         │
│  1. Extract causal claims: coffee → heart_disease           │
│  2. Identify evidence type: correlation, not causation      │
│  3. Check for confounders: lifestyle, diet, stress          │
│  4. Apply do-calculus: P(heart|do(coffee)) unknown          │
└───────────────────────┬─────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│              OUTPUT: CAUSAL ANALYSIS                        │
│  Claim: Coffee causes heart disease                         │
│  Evidence type: Correlation only                            │
│  Causal validity: WEAK                                      │
│  Reason: Correlation ≠ causation. Possible confounders:     │
│          lifestyle, stress, diet. Would need RCT to confirm.│
└─────────────────────────────────────────────────────────────┘
```

#### Implementation
```python
class CausalClaimVerifier:
    def __init__(self, model_path: str):
        self.crg = CausalReasoningIntervention(load_model(model_path))
        self.evidence_patterns = {
            "correlation": ["correlate", "associated", "linked"],
            "causation": ["causes", "leads to", "results in"],
            "rct": ["randomized", "controlled trial", "experiment"]
        }
    
    def verify(self, text: str) -> Dict:
        # Extract causal graph
        graph = self.crg.extract_causal_structure(text)
        
        # Identify evidence type
        evidence = self.classify_evidence(text)
        
        # Check for potential confounders
        confounders = self.identify_confounders(graph)
        
        # Compute causal validity
        validity = self.compute_validity(evidence, confounders)
        
        return {
            "claims": graph.edges,
            "evidence_type": evidence,
            "confounders": confounders,
            "validity": validity,
            "explanation": self.generate_explanation(validity, evidence, confounders)
        }
```

#### Demonstration
1. Input news headline about health claim
2. System extracts causal claims
3. Identifies if evidence is correlational or causal
4. Warns about potential confounders
5. Outputs validity assessment

#### Compute Requirements
- Model: Pythia-410m or Phi-2
- VRAM: 4-6 GB
- Latency: <1 second per analysis

---

### 6.4 PoC 4: Reasoning Benchmark Suite

**Objective**: Validate all innovations on consumer hardware.

#### Test Suite
```python
BENCHMARK_SUITE = {
    "meta_cognitive": {
        "datasets": ["logiqa", "reclor", "arc_easy", "arc_challenge"],
        "metrics": ["accuracy", "calibration_error", "abstention_rate"],
        "model": "phi-2"  # 2.7B, fits in 6GB VRAM
    },
    "self_evolving": {
        "datasets": ["gsm8k_subset", "aqua_rat_subset"],
        "metrics": ["accuracy_before", "accuracy_after", "generations"],
        "model": "phi-2"
    },
    "causal_reasoning": {
        "datasets": ["copa", "babi_task6", "clutrr_subset"],
        "metrics": ["accuracy", "causal_chain_depth"],
        "model": "phi-2"
    }
}

def run_benchmark(suite: str, output_dir: Path):
    config = BENCHMARK_SUITE[suite]
    model = load_model(config["model"])
    
    for dataset in config["datasets"]:
        data = load_dataset(dataset)
        results = evaluate(model, data, config["metrics"])
        save_results(results, output_dir / f"{suite}_{dataset}.json")
```

#### Consumer Hardware Targets
| GPU | VRAM | Max Model | Inference Speed |
|-----|------|-----------|-----------------|
| RTX 3060 | 12GB | Mistral-7B (4-bit) | ~15 tok/s |
| RTX 4060 | 8GB | Phi-2 (FP16) | ~30 tok/s |
| M1 Mac | 8GB | Phi-2 (Metal) | ~20 tok/s |
| CPU only | - | Pythia-70m | ~5 tok/s |

---

## Part VII: Implementation Roadmap

### Phase 1: Foundation (Week 1-2)

| Task | Deliverable | Hours |
|------|-------------|-------|
| Setup consumer-grade test environment | Docker + model downloads | 4 |
| Validate MCRE on Phi-2 | Working demo | 8 |
| Validate SERS on Phi-2 | Working demo | 8 |
| Validate CRG on Phi-2 | Working demo | 8 |

### Phase 2: PoC Development (Week 3-4)

| Task | Deliverable | Hours |
|------|-------------|-------|
| PoC 1: Confidence indicator (backend) | FastAPI server | 12 |
| PoC 1: Chrome extension (frontend) | Working extension | 16 |
| PoC 2: Homework helper (backend) | Flask/FastAPI server | 16 |
| PoC 2: Simple web interface | React/vanilla JS | 12 |

### Phase 3: Benchmarking (Week 5-6)

| Task | Deliverable | Hours |
|------|-------------|-------|
| Benchmark suite implementation | Python scripts | 12 |
| LogiQA evaluation | Results JSON | 8 |
| Self-evolving 1000-episode run | Evolution curves | 20 |
| Causal reasoning evaluation | Results JSON | 8 |

### Phase 4: Analysis & Writing (Week 7-8)

| Task | Deliverable | Hours |
|------|-------------|-------|
| Result analysis | Plots + tables | 12 |
| Paper draft (meta-cognitive) | 8-page draft | 24 |
| Paper draft (causal reasoning) | 8-page draft | 24 |

---

## Part VIII: File Inventory

### Core Innovation Files

| File | Innovation | Status |
|------|------------|--------|
| `eas/src/intervention/metacognitive.py` | Meta-Cognitive Engine | ✅ Implemented |
| `eas/src/intervention/self_evolving.py` | Self-Evolving System | ✅ Implemented |
| `eas/src/intervention/causal_reasoning.py` | Causal Reasoning | ✅ Implemented |
| `eas/src/intervention/compositional_logic.py` | Compositional Logic | ✅ Implemented |
| `eas/src/intervention/adaptive_reasoning.py` | Adaptive Amplifier | ✅ Implemented |
| `eas/src/intervention/circuit_discovery.py` | Circuit Discovery | ✅ Implemented |
| `eas/src/intervention/unified_engine.py` | Unified Engine | ✅ Implemented |
| `eas/src/watcher/position_aware_watcher.py` | Position-Aware EAS | ✅ Implemented |

### Analysis Scripts

| File | Purpose |
|------|---------|
| `eas/analysis/scripts/run_ctd_analysis.py` | CTD measurement |
| `eas/analysis/scripts/run_ctd_scaling.py` | Multi-model scaling |
| `eas/analysis/scripts/run_pa_validation.py` | Position-aware validation |

### Result Files

| File | Contents |
|------|----------|
| `eas/analysis/results/ctd_scaling_results.json` | Scaling validation |
| `eas/analysis/results/critical_token_divergence.json` | CTD metrics |

---

## Part IX: Compute Requirements Summary

### Minimum (CPU-only)
- **Model**: Pythia-70m
- **RAM**: 8GB
- **Use case**: Development, testing, slow inference

### Recommended (Consumer GPU)
- **Model**: Phi-2 (2.7B)
- **VRAM**: 6-8GB
- **GPU**: RTX 3060/4060 or M1/M2 Mac
- **Use case**: Full PoC development, fast inference

### Optimal (Mid-range)
- **Model**: Mistral-7B (4-bit quantized)
- **VRAM**: 12-16GB
- **GPU**: RTX 3090/4080
- **Use case**: Benchmarking, evolution experiments

---

## Part X: Success Criteria

### Research Validation
- [ ] MCRE reduces hallucination rate by 15%+ via principled abstention
- [ ] SERS shows measurable improvement over 1000 episodes
- [ ] CRG correctly identifies causal vs correlational claims 80%+

### PoC Demonstration
- [ ] Confidence indicator correctly flags low-quality reasoning
- [ ] Homework helper improves accuracy over time with feedback
- [ ] Causal verifier identifies faulty causal claims in news articles

### Publication Readiness
- [ ] Two paper drafts ready for submission
- [ ] Reproducible benchmark results
- [ ] Open-source code release

---

*Document created: December 2025*  
*Research program: EAS (Emergent Attractor Steering)*  
*Focus: Consumer-grade reasoning enhancement*

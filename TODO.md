# Experimental Framework for Evaluating Emergent Activation Snapping (EAS)

## 1. Core Components Implementation

### 1.1 Base Neural Network Implementation
- [ ] Implement minimal autoregressive transformer (2 layers, 8 heads, 512 hidden dim)
- [ ] Implement tokenizer specialized for logic/syllogism corpora (~500 tokens)
- [ ] Add hook interface to intercept and modify hidden state tensor at middle layer (Layer 1)
- [ ] Ensure weights are completely frozen after initial pre-training
- [ ] Implement model serialization/deserialization for frozen base model

### 1.2 Emergent Watcher Implementation
- [ ] Implement Attractor Memory (dynamic tensor A ∈ R^(K×D), default K=10)
- [ ] Implement Whitening Buffer (running statistics for normalization)
- [ ] Implement Clustering Engine (online K-Means algorithm)
- [ ] Implement signal preprocessing (pooling and whitening)
- [ ] Implement adaptive snapping mechanism with dynamic alpha and safety clamping
- [ ] Implement attractor evolution (conditional updates based on successful outcomes)
- [ ] Implement Sandwich Normalization to maintain hypersphere consistency

### 1.3 Advanced Watcher Features
- [ ] Implement Symbolic Verifier (mini-SAT logical checker)
- [ ] Implement Hyperbolic Geometry option (Poincaré ball arithmetic, Möbius addition)
- [ ] Implement attention-weighted pooling option as alternative to mean pooling

## 2. Data Generation and Management

### 2.1 Synthetic Logic Corpus
- [ ] Generate 1,000 diverse syllogism samples for pre-training
  - [ ] Include classic syllogisms (All X are Y. Z is X. -> Z is Y)
  - [ ] Include propositional logic (If P then Q. P. -> Q)
  - [ ] Include more complex logical constructs to test limits
- [ ] Generate 200 diverse samples for online evaluation with balanced difficulty
- [ ] Implement logic verification oracle for accurate labeling
- [ ] Create validation set for hyperparameter tuning
- [ ] Document logical types in dataset (transitivity, negation, conjunction, etc.)

### 2.2 Real Small-Scale Datasets
- [ ] Curate 100 real syllogism reasoning examples from cognitive science studies
- [ ] Curate 150 samples from small logical reasoning benchmarks
- [ ] Implement loading and preprocessing for real datasets
- [ ] Create validation protocol for real dataset performance
- [ ] Document differences between synthetic and real reasoning patterns

### 2.3 Dataset Analysis Tools
- [ ] Implement dataset difficulty metrics
- [ ] Analyze logical complexity distribution
- [ ] Create tokenization validation tools
- [ ] Compare synthetic vs real dataset characteristics

## 3. Implementation and Integration

### 3.1 Base Model Architecture Implementation
- [ ] Implement minimal autoregressive transformer (2 layers, 8 heads, 512 hidden dim)
- [ ] Implement smaller model variant (1 layer, 4 heads, 128 hidden dim)
- [ ] Implement tokenizer specialized for logic/syllogism corpora (~500 tokens)
- [ ] Add hook interface to intercept and modify hidden state tensor at middle layer (Layer 1)
- [ ] Ensure weights are completely frozen after initial pre-training
- [ ] Implement model serialization/deserialization for frozen base model
- [ ] Add proper configuration management for all hyperparameters
- [ ] Implement configurable model architecture (layers, heads, dimensions)
- [ ] Implement causal/bidirectional attention mechanism
- [ ] Implement GELU activation functions
- [ ] Implement layer normalization (pre-norm configuration)
- [ ] Implement residual connections
- [ ] Add dropout functionality (train vs inference mode)
- [ ] Create model validation to ensure architecture meets specifications
- [ ] Add model size configuration options for progressive experimentation

### 3.2 Base Model Training
- [ ] Train base model to 60-70% accuracy on pre-training set
- [ ] Verify model is properly frozen after training
- [ ] Test base model inference without Watcher intervention
- [ ] Profile base model inference time and memory usage
- [ ] Validate correct hook placement and functionality
- [ ] Implement training with proper validation splits
- [ ] Add early stopping based on validation performance

### 3.3 Watcher Integration
- [ ] Integrate Watcher with base model
- [ ] Implement online learning loop with intervention
- [ ] Ensure proper gradient blocking through Watcher to base model
- [ ] Test Watcher initialization with random normal attractors
- [ ] Validate signal flow between components
- [ ] Implement PyTorch forward hooks for Layer 1 activation capture
- [ ] Create proper activation modification pipeline (capture-modify-inject)
- [ ] Add validation that hook correctly intercepts and modifies activations
- [ ] Test gradient flow to ensure base model remains frozen during EAS

## 5. Experimental Protocol

### 5.1 Progressive Validation and Screening Phase
- [ ] Implement quick diagnostic run with small model (first 20 iterations) to assess basic EAS efficacy
- [ ] Pre-screen base model's activation clustering properties before full EAS run
- [ ] Run multiple hyperparameter combinations in parallel quick tests
- [ ] Implement statistical significance checks on early results to determine viability
- [ ] Create threshold checking functions for early stopping criteria
- [ ] Implement trend analysis for early prediction of experiment success
- [ ] Validate results on real dataset with small model before scaling up
- [ ] Compare efficiency metrics between small and standard models

### 5.2 Primary Evaluation Loop
- [ ] Implement main evaluation loop (200 iterations as specified)
  - Forward pass with Watcher.snap()
  - Check correctness via oracle
  - Conditional Watcher.update() on success
  - Log metrics at each iteration
- [ ] Implement efficient logging infrastructure (essential metrics only)
- [ ] Implement in-memory processing instead of checkpointing
- [ ] Add option for state reconstruction instead of saving full checkpoints
- [ ] Create selective metric logging (configurable detail level)
- [ ] Add experimental results aggregation without full state saves

### 5.3 Baseline Conditions
- [ ] Baseline: No Watcher (base model only)
- [ ] Random Control: Watcher enabled but update() disabled (static random attractors)
- [ ] Fixed Steering: Constant alpha value (no adaptive strength)
- [ ] No Clamping: Without safety magnitude clamping
- [ ] Euclidean Geometry: Without hyperbolic options
- [ ] Small model baselines: All baseline conditions tested with smaller model first

### 5.4 Rapid Feedback and Adaptive Mechanisms
- [ ] Implement early stopping criteria (e.g., no 20% improvement within first 50 updates)
- [ ] Create progressive complexity testing framework (start with simple problems)
- [ ] Build real-time monitoring dashboard for continuous metric updates
- [ ] Implement graduated intervention intensity (ramp up alpha values gradually)
- [ ] Create quick diagnostic tests (10-20 iterations) for parameter screening
- [ ] Implement failure detection heuristics with automated alerts
- [ ] Create pre-assessment validation of geometric clustering in base model
- [ ] Add statistical significance checks during experiment to detect early trends
- [ ] Implement adaptive hyperparameter adjustment based on performance trends

### 5.5 Scalability and Progressive Testing
- [ ] Test with smaller model (50K parameters) before standard model
- [ ] Test with varying logical problem complexity levels
- [ ] Test with different types of logical reasoning (not just syllogisms)
- [ ] Implement gradual intervention strength ramping to detect optimal thresholds
- [ ] Run experiments with multiple random seeds for statistical significance
- [ ] Compare efficiency and effectiveness between model sizes
- [ ] Document performance scaling between small and standard models

### 5.6 Real Dataset Validation
- [ ] Validate EAS performance on real syllogism reasoning dataset
- [ ] Compare synthetic vs real dataset performance
- [ ] Assess generalization from synthetic to real data
- [ ] Test transfer learning from synthetic to real datasets
- [ ] Document any differences in attractor formation between synthetic and real data

### 5.7 Hyperparameter Experiments
- [ ] Different attractor counts (K=5, 10, 15, 20)
- [ ] Different base alpha values (α=0.1, 0.3, 0.5, 0.7)
- [ ] Different max delta values (δ=0.1, 0.3, 0.5, 1.0)
- [ ] Different update batch sizes (N=3, 5, 10)
- [ ] Different pooling strategies (mean vs attention-weighted)

## 6. Comprehensive Evaluation Metrics

### 6.1 Primary Metrics
- [ ] Online learning curve: Accuracy over time (must improve ≥20% over baseline within 50-100 updates)
- [ ] Attractor Stability: Centroid variance (Euclidean shift per update) convergence to < 0.05
- [ ] Latency Overhead: Total inference time increase < 5%

### 6.2 Validation Requirements (Critical for experiment worthiness)
- [ ] Geometric Consistency Analysis: Verify that successful inferences actually cluster in activation space (validate core EAS assumption)
- [ ] Causality Testing: Implement ablation studies to ensure intervention causes improvement, not just correlation
- [ ] Stability Monitoring: Track system stability with controlled intervention parameters
- [ ] Generalizability Assessment: Evaluate improvement across different types of logical reasoning problems
- [ ] Attractor Formation Analysis: Validate that attractors form meaningful geometric structures rather than random patterns

### 6.3 Robustness & Safety Metrics
- [ ] Collapse Detection: Entropy of attractor usage, detect >80% snaps to single attractor
- [ ] Hallucination Rate: Monitor "off-manifold" drifts (distance threshold > 1.0)
- [ ] Intervention Frequency: Track how often snapping occurs
- [ ] Adaptive Alpha Distribution: Histogram of alpha values over time
- [ ] Attractor Activation Patterns: Which attractors are used for which logical types

### 6.4 Failure Mode Detection
- [ ] Monitor for divergence / destabilization
- [ ] Detect if accuracy degrades below baseline
- [ ] Track attractor drift velocity
- [ ] Monitor for mode collapse
- [ ] Detect overfitting to specific logical patterns

### 6.5 Secondary Metrics
- [ ] Attractor utilization rate (how evenly used are the attractors)
- [ ] Convergence speed of online K-means
- [ ] Memory usage of Watcher module
- [ ] Distribution of cosine similarities between activations and attractors

## 7. Visualization and Analysis

### 7.1 Dynamic Visualizations
- [ ] Real-time attractor evolution visualization
- [ ] t-SNE plots of activation trajectories
- [ ] Heatmap of attractor-logical type associations
- [ ] Learning curve comparison across conditions
- [ ] Real-time dashboard for monitoring experiment progress and early stopping
- [ ] Live trend analysis visualization for early success prediction
- [ ] Interactive parameter adjustment during experiment (if needed)
- [ ] Real-time resource usage monitoring (memory, CPU, time)
- [ ] Efficiency metrics dashboard (storage, computation time, memory usage)

### 7.2 Static Analysis Tools
- [ ] Attractor clustering analysis
- [ ] Performance vs. time plots
- [ ] Attractor stability over time
- [ ] Dimensionality reduction of attractor space

## 8. Failure Analysis and Contingency Plans

### 8.1 Potential Failure Modes
- [ ] No accuracy improvement over baseline
- [ ] System instability / divergence
- [ ] Mode collapse (all attractors converge to same point)
- [ ] Overfitting to specific logical types
- [ ] Excessive latency overhead
- [ ] Attractor cycling / unstable updates
- [ ] Insufficient attractor diversity
- [ ] Core assumption invalid: successful inferences do not cluster geometrically
- [ ] Intervention does not cause improvement (correlation without causation)
- [ ] System fails to remain stable with controlled parameters
- [ ] Slow convergence requiring more iterations than available budget
- [ ] Parameter configurations that lead to immediate failure
- [ ] Unreliable results across multiple random seeds

### 8.2 Mitigation Strategies
- [ ] Early stopping if performance degrades
- [ ] Attractor diversity regularization
- [ ] Dynamic adjustment of alpha based on stability
- [ ] Alternative clustering algorithms if K-means fails
- [ ] Reduced intervention frequency if unstable
- [ ] Multiple random seeds for statistical significance
- [ ] Graduated intervention intensity to find stability threshold
- [ ] Regular validation of geometric clustering assumption during runtime
- [ ] Adaptive parameter tuning based on performance trends
- [ ] Automatic detection of failed parameter configurations
- [ ] Dynamic allocation of computational budget based on early performance indicators

### 8.3 Alternative Approaches to Explore if Primary Fails
- [ ] Supervised attractor initialization (pre-trained on logical problems)
- [ ] Different geometric spaces (Riemannian, hyperbolic)
- [ ] Hierarchical attractor structure
- [ ] Multi-layer intervention (not just middle layer)
- [ ] Different normalization strategies
- [ ] Reinforcement learning approach to attractor updates
- [ ] Attention-based attractor selection instead of cosine similarity
- [ ] Supervised steering vector approach as fallback
- [ ] Hybrid symbolic-neural integration approach
- [ ] Gradient-based fine-tuning of specific reasoning components
- [ ] Rapid assessment techniques to quickly determine if approach is promising
- [ ] Lightweight proxy tasks for fast evaluation of intervention effectiveness

## 9. Extended Experiments

### 9.1 Transfer Learning Tests
- [ ] Test EAS trained on one logical domain applied to another
- [ ] Cross-validation with different logical problem types

### 9.2 Scalability Analysis
- [ ] Test with larger base models
- [ ] Test with more complex logical problems
- [ ] Performance at different computational budgets

### 9.3 Comparative Analysis
- [ ] Comparison with supervised steering vectors
- [ ] Comparison with explicit symbolic encoders
- [ ] Comparison with fine-tuning approaches
- [ ] Ablation studies for different EAS components

## 10. Documentation and Reproducibility

### 10.1 Experimental Documentation
- [ ] Complete specification of all hyperparameters
- [ ] Detailed logging of all experimental conditions
- [ ] Code documentation for all components
- [ ] Reproducibility guidelines

### 10.2 Efficiency and Performance Analysis
- [ ] Compare memory usage between checkpoint and in-memory approaches
- [ ] Measure time savings from eliminated checkpointing
- [ ] Document performance differences between small and standard models
- [ ] Analyze storage reduction from selective logging
- [ ] Track computational efficiency improvements

### 10.3 Results Analysis
- [ ] Statistical significance testing
- [ ] Confidence intervals for metrics
- [ ] Effect size calculations
- [ ] Cross-validation results

## 11. System Setup and Dependencies

### 11.1 Environment Configuration
- [ ] Create requirements.txt with specific package versions
- [ ] Implement environment validation script
- [ ] Document system requirements (RAM, disk space, etc.)
- [ ] Create Dockerfile for reproducible environment
- [ ] Add configuration management for different hardware setups

### 11.2 Resource Management
- [ ] Implement memory usage monitoring
- [ ] Add GPU/CPU selection configuration
- [ ] Create resource estimation tools
- [ ] Implement graceful degradation when resources are limited
- [ ] Add memory-efficient processing options
- [ ] Optimize data loading for minimal memory footprint
- [ ] Implement streaming data processing

## 12. Optional Extensions (Priority: Lower)

### 12.1 Advanced Features
- [ ] Implement symbolic verifier integration
- [ ] Implement hyperbolic geometry option
- [ ] Multi-head attention in transformer if time permits
- [ ] Additional visualization techniques

### 12.2 Additional Baselines
- [ ] Comparison with gradient-based fine-tuning
- [ ] Comparison with other intervention methods
- [ ] Zero-shot performance baselines
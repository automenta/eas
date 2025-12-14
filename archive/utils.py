#!/usr/bin/env python3
"""
utils.py - Shared utilities for David vs Goliath benchmark

Includes:
- Model loading helpers (with device detection)
- Dataset loading and formatting
- Evaluation functions
- Metrics calculation
- Result visualization
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from tqdm import tqdm
import json
import time


# Model registry with sizes and layer counts
MODEL_REGISTRY = {
    "pythia-70m": {
        "name": "EleutherAI/pythia-70m",
        "size": "70M",
        "hidden_dim": 512,
        "num_layers": 6,
        "category": "david"
    },
    "pythia-160m": {
        "name": "EleutherAI/pythia-160m", 
        "size": "160M",
        "hidden_dim": 768,
        "num_layers": 12,
        "category": "david"
    },
    "pythia-410m": {
        "name": "EleutherAI/pythia-410m",
        "size": "410M", 
        "hidden_dim": 1024,
        "num_layers": 24,
        "category": "goliath"
    },
    "gpt2": {
        "name": "gpt2",
        "size": "117M",
        "hidden_dim": 768,
        "num_layers": 12,
        "category": "david"
    },
    "gpt2-medium": {
        "name": "gpt2-medium",
        "size": "345M",
        "hidden_dim": 1024,
        "num_layers": 24,
        "category": "goliath"
    },
    "gpt2-large": {
        "name": "gpt2-large",
        "size": "774M",
        "hidden_dim": 1280,
        "num_layers": 36,
        "category": "goliath"
    },
}


def get_device() -> str:
    """Detect best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(model_key: str, device: Optional[str] = None) -> Tuple[Any, Any, Dict]:
    """
    Load model and tokenizer from registry.
    
    Returns:
        (model, tokenizer, model_info)
    """
    if model_key not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_key}. Available: {list(MODEL_REGISTRY.keys())}")
    
    info = MODEL_REGISTRY[model_key]
    device = device or get_device()
    
    print(f"Loading {model_key} ({info['size']}) on {device}...")
    
    tokenizer = AutoTokenizer.from_pretrained(info["name"])
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        info["name"],
        torch_dtype=torch.float32,  # Use float32 for CPU compatibility
    ).to(device)
    model.eval()
    
    return model, tokenizer, info


@dataclass
class EvalSample:
    """Single evaluation sample."""
    question: str
    options: List[str]
    correct_idx: int
    prompt: str = ""
    
    def __post_init__(self):
        if not self.prompt:
            options_text = "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(self.options)])
            self.prompt = f"Question: {self.question}\n{options_text}\nAnswer:"


def load_logiqa(split: str = "validation", max_samples: Optional[int] = None) -> List[EvalSample]:
    """Load reasoning dataset formatted for evaluation.
    
    Uses Hellaswag by default as it has 10K+ samples for proper benchmarking.
    """
    # Use Hellaswag for reasoning - has 10K validation samples
    return load_hellaswag(split, max_samples)


def load_hellaswag(split: str = "validation", max_samples: Optional[int] = None) -> List[EvalSample]:
    """Load Hellaswag dataset as fallback for reasoning evaluation."""
    print(f"Loading Hellaswag {split} split...")
    dataset = load_dataset("Rowan/hellaswag", split=split, trust_remote_code=True)
    
    samples = []
    for i, ex in enumerate(dataset):
        if max_samples and i >= max_samples:
            break
        
        # Hellaswag: ctx (context), endings (4 options), label (0-3)
        samples.append(EvalSample(
            question=ex["ctx"],
            options=ex["endings"],
            correct_idx=int(ex["label"])
        ))
    
    print(f"Loaded {len(samples)} samples")
    return samples


def load_arc_challenge(split: str = "validation", max_samples: Optional[int] = None) -> List[EvalSample]:
    """Load ARC-Challenge dataset formatted for evaluation."""
    print(f"Loading ARC-Challenge {split} split...")
    dataset = load_dataset("allenai/ai2_arc", "ARC-Challenge", split=split)
    
    samples = []
    for i, ex in enumerate(dataset):
        if max_samples and i >= max_samples:
            break
        
        # ARC uses letter labels, convert to index
        choices = ex["choices"]
        options = choices["text"]
        labels = choices["label"]
        correct_label = ex["answerKey"]
        
        try:
            correct_idx = labels.index(correct_label)
        except ValueError:
            continue  # Skip malformed samples
        
        samples.append(EvalSample(
            question=ex["question"],
            options=options,
            correct_idx=correct_idx
        ))
    
    print(f"Loaded {len(samples)} samples")
    return samples


def get_answer_probs(
    model: Any,
    tokenizer: Any,
    prompt: str,
    num_options: int = 4,
    device: str = "cpu"
) -> Tuple[List[float], int, float]:
    """
    Get model's probability distribution over answer choices.
    
    Returns:
        (probs, predicted_idx, max_prob)
    """
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0, -1, :]  # Last token
    
    # Get probabilities for answer tokens (A, B, C, D, ...)
    probs = []
    for i in range(num_options):
        letter = chr(65 + i)  # A, B, C, D
        # Try with space prefix (more common)
        token_ids = tokenizer.encode(f" {letter}", add_special_tokens=False)
        if token_ids:
            token_id = token_ids[0]
        else:
            token_ids = tokenizer.encode(letter, add_special_tokens=False)
            token_id = token_ids[0] if token_ids else 0
        
        prob = F.softmax(logits, dim=-1)[token_id].item()
        probs.append(prob)
    
    # Normalize to sum to 1
    total = sum(probs)
    if total > 0:
        probs = [p / total for p in probs]
    
    predicted_idx = max(range(len(probs)), key=lambda i: probs[i])
    max_prob = max(probs)
    
    return probs, predicted_idx, max_prob


@dataclass
class EvalResult:
    """Result of evaluating a single sample."""
    correct: bool
    predicted_idx: int
    correct_idx: int
    confidence: float
    probs: List[float]


@dataclass 
class BenchmarkResult:
    """Aggregate benchmark results."""
    model_name: str
    config_name: str
    num_samples: int
    num_correct: int
    accuracy: float
    avg_confidence: float
    avg_confidence_correct: float
    avg_confidence_wrong: float
    total_time: float
    per_sample_time: float
    extra_stats: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "model": self.model_name,
            "config": self.config_name,
            "samples": self.num_samples,
            "correct": self.num_correct,
            "accuracy": f"{self.accuracy:.1%}",
            "avg_confidence": f"{self.avg_confidence:.3f}",
            "time_per_sample": f"{self.per_sample_time:.3f}s",
            **self.extra_stats
        }


def evaluate_model(
    model: Any,
    tokenizer: Any,
    samples: List[EvalSample],
    device: str = "cpu",
    show_progress: bool = True,
    intervener: Optional[Any] = None,  # EASIntervener
) -> BenchmarkResult:
    """
    Evaluate model on samples and return results.
    
    If intervener is provided, updates attractors on correct predictions.
    """
    results: List[EvalResult] = []
    start_time = time.time()
    
    iterator = tqdm(samples, desc="Evaluating") if show_progress else samples
    
    for sample in iterator:
        probs, pred_idx, confidence = get_answer_probs(
            model, tokenizer, sample.prompt, 
            num_options=len(sample.options),
            device=device
        )
        
        correct = (pred_idx == sample.correct_idx)
        
        results.append(EvalResult(
            correct=correct,
            predicted_idx=pred_idx,
            correct_idx=sample.correct_idx,
            confidence=confidence,
            probs=probs
        ))
        
        # Update EAS on correct predictions
        if intervener is not None:
            intervener.record_sample()
            if correct:
                intervener.update_on_success()
    
    total_time = time.time() - start_time
    
    # Calculate metrics
    num_correct = sum(1 for r in results if r.correct)
    accuracy = num_correct / len(results) if results else 0
    
    correct_confs = [r.confidence for r in results if r.correct]
    wrong_confs = [r.confidence for r in results if not r.correct]
    
    avg_conf = sum(r.confidence for r in results) / len(results) if results else 0
    avg_conf_correct = sum(correct_confs) / len(correct_confs) if correct_confs else 0
    avg_conf_wrong = sum(wrong_confs) / len(wrong_confs) if wrong_confs else 0
    
    extra_stats = {}
    if intervener is not None:
        stats = intervener.get_stats()
        extra_stats["eas_interventions"] = stats["intervention_count"]
        extra_stats["eas_entropy"] = f"{stats['attractor_entropy']:.3f}"
    
    return BenchmarkResult(
        model_name="",  # Set by caller
        config_name="",  # Set by caller
        num_samples=len(results),
        num_correct=num_correct,
        accuracy=accuracy,
        avg_confidence=avg_conf,
        avg_confidence_correct=avg_conf_correct,
        avg_confidence_wrong=avg_conf_wrong,
        total_time=total_time,
        per_sample_time=total_time / len(results) if results else 0,
        extra_stats=extra_stats
    )


def print_results_table(results: List[BenchmarkResult]):
    """Print formatted results table."""
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)
    
    # Header
    headers = ["Model", "Config", "Accuracy", "Confidence", "Time/Sample"]
    row_format = "{:<20} {:<15} {:<12} {:<12} {:<12}"
    print(row_format.format(*headers))
    print("-" * 80)
    
    # Sort by accuracy
    sorted_results = sorted(results, key=lambda r: r.accuracy, reverse=True)
    
    for r in sorted_results:
        print(row_format.format(
            r.model_name[:20],
            r.config_name[:15],
            f"{r.accuracy:.1%}",
            f"{r.avg_confidence:.3f}",
            f"{r.per_sample_time:.3f}s"
        ))
    
    print("=" * 80)
    
    # Find best
    best = sorted_results[0]
    print(f"\n🏆 Best: {best.model_name} + {best.config_name} with {best.accuracy:.1%} accuracy")


def save_results(results: List[BenchmarkResult], filepath: str = "results.json"):
    """Save results to JSON file."""
    data = [r.to_dict() for r in results]
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Results saved to {filepath}")


if __name__ == "__main__":
    # Quick test
    print("Utils Module - Quick Test")
    print("=" * 50)
    
    # Test device detection
    device = get_device()
    print(f"Detected device: {device}")
    
    # Test dataset loading
    samples = load_logiqa(max_samples=5)
    print(f"Sample prompt:\n{samples[0].prompt[:200]}...")
    
    print("\n✅ Utils module working correctly!")

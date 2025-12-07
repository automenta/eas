"""
Metrics Tracking and Logging for EAS Experiment
Implements comprehensive evaluation metrics and logging infrastructure
"""
import json
import time
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime
import os


class MetricsTracker:
    """Comprehensive metrics tracker for EAS experiments"""
    
    def __init__(self, experiment_name: str = "EAS_Experiment"):
        self.experiment_name = experiment_name
        self.start_time = time.time()
        self.metrics = {
            'accuracy': [],
            'latency': [],
            'intervention_frequency': [],
            'attractor_stability': [],
            'snap_history': [],
            'entropy': [],
            'hallucination_rate': [],
            'convergence_speed': [],
            'attractor_utilization': [],
            'cosine_similarities': [],
            'delta_magnitude': [],
            'alpha_values': []
        }
        self.experiment_config = {}
        self.results_summary = {}
    
    def add_metric(self, metric_name: str, value: Any):
        """Add a value to a specific metric."""
        if metric_name not in self.metrics:
            self.metrics[metric_name] = []
        self.metrics[metric_name].append(value)
    
    def update_metrics(self, new_metrics: Dict[str, Any]):
        """Update multiple metrics at once."""
        for name, value in new_metrics.items():
            self.add_metric(name, value)
    
    def calculate_entropy(self, attractor_usage: List[int], num_attractors: int) -> float:
        """Calculate entropy of attractor usage (0 = all same attractor, high = evenly distributed)."""
        if not attractor_usage:
            return 0.0
        
        unique, counts = np.unique(attractor_usage, return_counts=True)
        probs = counts / len(attractor_usage)
        entropy = -np.sum(probs * np.log2(probs + 1e-8))
        max_entropy = np.log2(num_attractors)
        
        return entropy / max_entropy if max_entropy > 0 else 0.0
    
    def calculate_hallucination_rate(self, original_activations: List, snapped_activations: List) -> float:
        """Calculate rate of 'off-manifold' drifts."""
        if not original_activations or not snapped_activations:
            return 0.0
        
        distances = []
        for orig, snapped in zip(original_activations, snapped_activations):
            if hasattr(orig, 'cpu') and hasattr(snapped, 'cpu'):
                # PyTorch tensors
                dist = torch.norm(orig.cpu() - snapped.cpu(), p=2).item()
            else:
                # Numpy arrays
                dist = np.linalg.norm(orig - snapped)
            distances.append(dist)
        
        # Calculate what proportion of distances exceed safety threshold
        threshold = 1.0  # As specified in the requirements
        hallucination_count = sum(1 for d in distances if d > threshold)
        
        return hallucination_count / len(distances) if distances else 0.0
    
    def calculate_attractor_utilization(self, snap_history: List[int], num_attractors: int) -> List[float]:
        """Calculate utilization rate of each attractor."""
        if not snap_history:
            return [0.0] * num_attractors
        
        unique, counts = np.unique(snap_history, return_counts=True)
        utilization = [0.0] * num_attractors
        
        for idx, count in zip(unique, counts):
            if idx < num_attractors:
                utilization[idx] = count / len(snap_history)
        
        return utilization
    
    def generate_summary(self) -> Dict[str, Any]:
        """Generate a summary of the experiment results."""
        summary = {
            'experiment_name': self.experiment_name,
            'total_duration': time.time() - self.start_time,
            'final_accuracy': self.metrics['accuracy'][-1] if self.metrics['accuracy'] else 0,
            'accuracy_improvement': 0,
            'avg_latency': np.mean(self.metrics['latency']) if self.metrics['latency'] else 0,
            'latency_overhead': 0,  # This would need baseline comparison
            'final_attractor_stability': self.metrics['attractor_stability'][-1] if self.metrics['attractor_stability'] else 0,
            'collapse_detection': self.check_collapse(),
            'hallucination_rate': np.mean(self.metrics['hallucination_rate']) if self.metrics['hallucination_rate'] else 0,
            'entropy': np.mean(self.metrics['entropy']) if self.metrics['entropy'] else 0
        }
        
        # Calculate accuracy improvement if we have baseline
        if 'baseline_accuracy' in self.experiment_config:
            summary['accuracy_improvement'] = (
                summary['final_accuracy'] - self.experiment_config['baseline_accuracy']
            )
        
        # Add convergence information
        if len(self.metrics['accuracy']) > 10:
            early_acc = np.mean(self.metrics['accuracy'][:10])
            late_acc = np.mean(self.metrics['accuracy'][-10:])
            summary['convergence_rate'] = late_acc - early_acc
        
        self.results_summary = summary
        return summary
    
    def check_collapse(self) -> bool:
        """Check for mode collapse (>80% of snaps to single attractor)."""
        if not self.metrics['snap_history']:
            return False
        
        # Get the last N snaps to check for recent collapse
        recent_snaps = self.metrics['snap_history'][-50:] if len(self.metrics['snap_history']) >= 50 else self.metrics['snap_history']
        if not recent_snaps:
            return False
        
        unique, counts = np.unique(recent_snaps, return_counts=True)
        max_usage = max(counts) if counts.size > 0 else 0
        collapse_ratio = max_usage / len(recent_snaps)
        
        return collapse_ratio > 0.8
    
    def save_metrics(self, filepath: str):
        """Save metrics to a JSON file."""
        data = {
            'experiment_config': self.experiment_config,
            'metrics': self.metrics,
            'results_summary': self.results_summary,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    def load_metrics(self, filepath: str):
        """Load metrics from a JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
            self.experiment_config = data.get('experiment_config', {})
            self.metrics = data.get('metrics', {})
            self.results_summary = data.get('results_summary', {})
    
    def set_experiment_config(self, config: Dict[str, Any]):
        """Set the experiment configuration."""
        self.experiment_config = config


class EASLogger:
    """Logger for EAS experiments with real-time monitoring capabilities"""
    
    def __init__(self, log_dir: str = "logs", experiment_name: str = "EAS_Experiment"):
        self.log_dir = log_dir
        self.experiment_name = experiment_name
        self.metrics_tracker = MetricsTracker(experiment_name)
        
        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # Initialize log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(log_dir, f"{experiment_name}_{timestamp}.log")
        self.metrics_file = os.path.join(log_dir, f"{experiment_name}_{timestamp}_metrics.json")
    
    def log_message(self, message: str, level: str = "INFO"):
        """Log a message with timestamp."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {level}: {message}\n"
        
        with open(self.log_file, 'a') as f:
            f.write(log_entry)
    
    def log_hyperparameters(self, config: Dict[str, Any]):
        """Log all hyperparameters for reproducibility."""
        self.log_message("Experiment Configuration:")
        for key, value in config.items():
            self.log_message(f"  {key}: {value}")
        
        # Also update metrics tracker config
        self.metrics_tracker.set_experiment_config(config)
    
    def log_iteration(self, iteration: int, accuracy: float, latency: float, 
                     intervention_freq: float, attractor_stability: float):
        """Log metrics for a single iteration."""
        self.metrics_tracker.update_metrics({
            'accuracy': accuracy,
            'latency': latency,
            'intervention_frequency': intervention_freq,
            'attractor_stability': attractor_stability
        })
        
        if iteration % 10 == 0:  # Log every 10 iterations
            self.log_message(
                f"Iteration {iteration}: Accuracy={accuracy:.4f}, "
                f"Latency={latency:.4f}s, Interventions={intervention_freq}, "
                f"Attractor Stability={attractor_stability:.4f}"
            )
    
    def log_final_results(self):
        """Log final experiment results."""
        summary = self.metrics_tracker.generate_summary()
        
        self.log_message("=== EXPERIMENT RESULTS ===")
        for key, value in summary.items():
            self.log_message(f"{key}: {value}")
        
        # Save metrics to file
        self.metrics_tracker.save_metrics(self.metrics_file)
        
        self.log_message(f"Metrics saved to: {self.metrics_file}")
    
    def save_visualizations(self, output_dir: Optional[str] = None):
        """Save visualizations of the experiment results."""
        if output_dir is None:
            output_dir = self.log_dir

        os.makedirs(output_dir, exist_ok=True)

        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            # Set style for better-looking plots
            sns.set_style("whitegrid")
            plt.rcParams['figure.figsize'] = (12, 8)

            # 1. Accuracy over time
            if self.metrics_tracker.metrics['accuracy']:
                plt.figure(figsize=(10, 6))
                plt.plot(self.metrics_tracker.metrics['accuracy'])
                plt.title('Accuracy Over Time')
                plt.xlabel('Iteration')
                plt.ylabel('Accuracy')
                plt.grid(True)
                plt.savefig(os.path.join(output_dir, f"{self.experiment_name}_accuracy.png"))
                plt.close()

            # 2. Latency over time
            if self.metrics_tracker.metrics['latency']:
                plt.figure(figsize=(10, 6))
                plt.plot(self.metrics_tracker.metrics['latency'])
                plt.title('Latency Over Time')
                plt.xlabel('Iteration')
                plt.ylabel('Latency (seconds)')
                plt.grid(True)
                plt.savefig(os.path.join(output_dir, f"{self.experiment_name}_latency.png"))
                plt.close()

            # 3. Attractor stability
            if self.metrics_tracker.metrics['attractor_stability']:
                plt.figure(figsize=(10, 6))
                plt.plot(self.metrics_tracker.metrics['attractor_stability'])
                plt.title('Attractor Stability Over Time')
                plt.xlabel('Iteration')
                plt.ylabel('Stability (Entropy)')
                plt.grid(True)
                plt.savefig(os.path.join(output_dir, f"{self.experiment_name}_stability.png"))
                plt.close()

            # 4. Intervention frequency
            if self.metrics_tracker.metrics['intervention_frequency']:
                plt.figure(figsize=(10, 6))
                plt.plot(self.metrics_tracker.metrics['intervention_frequency'])
                plt.title('Intervention Frequency Over Time')
                plt.xlabel('Iteration')
                plt.ylabel('Number of Interventions')
                plt.grid(True)
                plt.savefig(os.path.join(output_dir, f"{self.experiment_name}_interventions.png"))
                plt.close()

            self.log_message(f"Visualizations saved to: {output_dir}")
        except ImportError:
            self.log_message("Matplotlib not available, skipping visualization generation")


def log_experiment_comparison(baseline_results: Dict, eas_results: Dict, logger: EASLogger):
    """Log comparison between baseline and EAS results."""
    logger.log_message("=== BASELINE vs EAS COMPARISON ===")
    
    # Compare final accuracies
    baseline_acc = baseline_results.get('accuracy', [])
    eas_acc = eas_results.get('accuracy', [])
    
    if baseline_acc and eas_acc:
        baseline_final = baseline_acc[-1]
        eas_final = eas_acc[-1]
        
        logger.log_message(f"Baseline Final Accuracy: {baseline_final:.4f}")
        logger.log_message(f"EAS Final Accuracy: {eas_final:.4f}")
        logger.log_message(f"Improvement: {eas_final - baseline_final:.4f}")
        
        # Check if EAS met the 20% improvement threshold
        improvement_threshold_met = (eas_final - baseline_final) >= 0.20
        logger.log_message(f"20% Improvement Threshold Met: {improvement_threshold_met}")
    
    # Compare other metrics
    if eas_results.get('latency', []):
        avg_latency = np.mean(eas_results['latency'])
        latency_overhead_acceptable = avg_latency < 0.05  # <5% overhead
        logger.log_message(f"Average Latency: {avg_latency:.4f}s")
        logger.log_message(f"Latency Threshold (<0.05s) Met: {latency_overhead_acceptable}")
    
    # Check stability metrics
    collapse_detected = logger.metrics_tracker.check_collapse()
    logger.log_message(f"Mode Collapse Detected: {collapse_detected}")
    
    # Check hallucination rate
    hallucination_rate = np.mean(eas_results.get('hallucination_rate', [0]))
    hallucination_acceptable = hallucination_rate < 0.2  # Assuming <20% is acceptable
    logger.log_message(f"Hallucination Rate: {hallucination_rate:.4f}")
    logger.log_message(f"Hallucination Rate Acceptable (<0.2): {hallucination_acceptable}")
import torch
import numpy as np
from eas.src.models.legacy.transformer_toy import AutoregressiveTransformer, create_standard_model
from eas.src.watcher import EmergentWatcher
from eas.src.models.legacy.tokenizer_toy import LogicTokenizer
from eas.advanced_validation.datasets import AvicennaLoader, ComplexLogicGenerator
import time
import json
import os
import re
import random

class AdvancedValidationSuite:
    def __init__(self, model_path=None, results_dir="eas/advanced_validation/results"):
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)

        # Defaults (will be overwritten by injection if using pretrained)
        self.tokenizer = LogicTokenizer(vocab_size=1500)
        self.is_pretrained = False

        # Initialize Model (Mock loading if path not provided)
        # In a real scenario, we load the pre-trained weights.
        self.model = create_standard_model(vocab_size=1500)
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

        # Check device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        # Initialize Watcher (default settings from README)
        # We'll re-initialize this properly when a model is injected
        self.watcher = EmergentWatcher(dim=512, k=10, alpha_base=0.3)
        self.watcher.to(self.device)
        self.watcher.attractor_memory.attractors.data = self.watcher.attractor_memory.attractors.data.to(self.device)

        # Data
        self.complex_gen = ComplexLogicGenerator()
        self.avicenna_loader = AvicennaLoader("eas/advanced_validation/data/avicenna_samples.json")

        # Metrics storage
        self.results = []

        # Pre-load Avicenna data for splitting
        self.avicenna_data = []
        raw_avicenna = self.avicenna_loader.load()
        for d in raw_avicenna:
            self.avicenna_data.append({
                "text": d['premise1'] + " " + d['premise2'],
                "target": d['label'],
                "type": "real"
            })

        # Split Avicenna (Simple 50/50 split for small dataset)
        random.seed(42) # Ensure repeatability
        random.shuffle(self.avicenna_data)
        split_idx = len(self.avicenna_data) // 2
        self.avicenna_warmup = self.avicenna_data[:split_idx]
        self.avicenna_test = self.avicenna_data[split_idx:]

    def _encode_sample(self, text):
        """Encodes text using tokenizer."""
        if self.is_pretrained:
            # Use HF tokenizer
            # Return tensor [1, seq_len]
            inputs = self.tokenizer(text, return_tensors="pt")
            return inputs["input_ids"].to(self.device)
        else:
            # Use Toy tokenizer
            tokens = self.tokenizer.tokenize(text)
            token_ids = []

            for w in tokens:
                if w in self.tokenizer.vocab:
                    token_ids.append(self.tokenizer.vocab[w])
                elif w.upper() in self.tokenizer.vocab:
                    token_ids.append(self.tokenizer.vocab[w.upper()])
                elif w.lower() in self.tokenizer.vocab:
                    token_ids.append(self.tokenizer.vocab[w.lower()])
                elif '<UNK>' in self.tokenizer.vocab:
                    token_ids.append(self.tokenizer.vocab['<UNK>'])
                else:
                    token_ids.append(1)

            return torch.tensor([token_ids], dtype=torch.long).to(self.device)

    def warmup_watcher(self, num_samples=100, dataset_type="synthetic"):
        """
        Supervised Warmup for the Watcher.
        dataset_type: 'synthetic' or 'avicenna'
        """
        print(f"Warming up Watcher with {num_samples} supervised samples ({dataset_type})...")

        # Disable intervention during warmup
        self.model.remove_intervention_hook(self.model.middle_layer_idx)

        data = []
        if dataset_type == "synthetic":
            data = self.complex_gen.generate_dataset(size=num_samples, distractors=False)
        elif dataset_type == "avicenna":
            data = self.avicenna_warmup
            if num_samples < len(data):
                 data = data[:num_samples]

        # Attach passive hook to capture activations
        def passive_hook(activations):
            return activations
        self.model.register_intervention_hook(self.model.middle_layer_idx, passive_hook)

        for sample in data:
            full_text = f"Question: {sample['text']}\nAnswer: {sample['target']}"
            input_tensor = self._encode_sample(full_text)

            with torch.no_grad():
                self.model(input_tensor)

            hidden = self.model.get_layer_activation(self.model.middle_layer_idx)

            if hidden is not None:
                self.watcher.update(hidden)

        print("Watcher warmup complete.")

    def run_scenario(self, scenario_name, dataset_name, intervention_type="none", num_samples=50):
        print(f"Running Scenario: {scenario_name} on {dataset_name} ({intervention_type})")

        # Prepare Data
        if dataset_name == "complex_synthetic":
            data = self.complex_gen.generate_dataset(size=num_samples, distractors=(intervention_type=="adversarial"))
        elif dataset_name == "avicenna":
            data = self.avicenna_test
            if num_samples < len(data):
                data = data[:num_samples]

        correct_count = 0
        latencies = []

        if intervention_type in ["standard", "adversarial"]:
            self.model.register_intervention_hook(self.model.middle_layer_idx, self.watcher.snap)
        else:
            self.model.remove_intervention_hook(self.model.middle_layer_idx)

        for sample in data:
            start_time = time.time()
            prompt = f"Question: {sample['text']}\nAnswer:"
            input_tensor = self._encode_sample(prompt)

            if input_tensor.size(1) > 128:
                input_tensor = input_tensor[:, -128:]

            with torch.no_grad():
                output = self.model(input_tensor)

            latencies.append(time.time() - start_time)

            next_token_logits = output[0, -1, :]
            next_token_id = torch.argmax(next_token_logits).item()
            decoded = self.tokenizer.decode([next_token_id])

            target = sample['target']
            is_correct = False

            target_first_token = target.split()[0]
            decoded_clean = decoded.strip().lower()
            target_clean = target_first_token.strip().lower()

            if target_clean in decoded_clean:
                is_correct = True
            elif decoded_clean in target_clean and len(decoded_clean) > 1:
                is_correct = True

            if target_clean in ['yes', 'true'] and decoded_clean in ['yes', 'true', 'correct']:
                is_correct = True
            if target_clean in ['no', 'false'] and decoded_clean in ['no', 'false', 'incorrect']:
                is_correct = True

            if is_correct:
                correct_count += 1
                hidden = self.model.get_layer_activation(self.model.middle_layer_idx)
                if intervention_type in ["standard", "adversarial"] and hidden is not None:
                    self.watcher.update(hidden)

        accuracy = correct_count / len(data) if data else 0
        avg_latency = sum(latencies) / len(latencies) if latencies else 0

        result = {
            "scenario": scenario_name,
            "dataset": dataset_name,
            "intervention": intervention_type,
            "accuracy": accuracy,
            "latency": avg_latency,
            "samples": len(data)
        }
        self.results.append(result)
        return result

    def reset_watcher(self):
        """Resets the watcher state for a new trial"""
        # Re-initialize the watcher to match the current model dimension
        if self.is_pretrained:
            dim = self.model.d_model
        else:
            dim = 512 # toy default

        self.watcher = EmergentWatcher(dim=dim, k=10, alpha_base=0.3)
        self.watcher.to(self.device)

    def run_multiple_trials(self, num_trials=5):
        """Runs the entire suite multiple times and aggregates results"""
        print(f"Starting {num_trials}-Trial Robustness Validation...")
        aggregated_results = {}

        for trial in range(num_trials):
            print(f"\n--- TRIAL {trial+1}/{num_trials} ---")
            self.reset_watcher()

            # Warmup Phase (Mix of Synthetic and Avicenna-Train)
            self.warmup_watcher(num_samples=50, dataset_type="synthetic")
            self.warmup_watcher(num_samples=20, dataset_type="avicenna")

            res_base_syn = self.run_scenario("Baseline", "complex_synthetic", "none", 50)
            res_base_avi = self.run_scenario("Baseline", "avicenna", "none", 20)

            res_eas_syn = self.run_scenario("EAS_Standard", "complex_synthetic", "standard", 50)
            res_eas_avi = self.run_scenario("EAS_Standard", "avicenna", "standard", 20)

            res_adv_syn = self.run_scenario("EAS_Adversarial", "complex_synthetic", "adversarial", 50)

            # Aggregate
            results_list = [res_base_syn, res_base_avi, res_eas_syn, res_eas_avi, res_adv_syn]

            for res in results_list:
                key = (res['scenario'], res['dataset'], res['intervention'])
                if key not in aggregated_results:
                    aggregated_results[key] = {'accuracy': [], 'latency': []}
                aggregated_results[key]['accuracy'].append(res['accuracy'])
                aggregated_results[key]['latency'].append(res['latency'])

        # Compute Statistics
        final_stats = []
        print("\n=== FINAL ROBUSTNESS REPORT ===")
        print(f"{'Scenario':<20} | {'Dataset':<20} | {'Mean Acc':<10} | {'Std Dev':<10} | {'Improvement'}")
        print("-" * 90)

        baseline_means = {}
        for key, vals in aggregated_results.items():
            if key[2] == 'none':
                baseline_means[key[1]] = np.mean(vals['accuracy'])

        for key, vals in aggregated_results.items():
            mean_acc = np.mean(vals['accuracy'])
            std_acc = np.std(vals['accuracy'])
            mean_lat = np.mean(vals['latency'])

            imp_str = "-"
            if key[2] != 'none':
                base_mean = baseline_means.get(key[1], 0)
                improvement = mean_acc - base_mean
                imp_str = f"{improvement:+.4f}"

            print(f"{key[0]:<20} | {key[1]:<20} | {mean_acc:.4f}     | {std_acc:.4f}     | {imp_str}")

            final_stats.append({
                "scenario": key[0],
                "dataset": key[1],
                "intervention": key[2],
                "mean_accuracy": mean_acc,
                "std_accuracy": std_acc,
                "mean_latency": mean_lat,
                "trials": num_trials
            })

        self.results = final_stats
        return final_stats

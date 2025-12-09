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
        self.watcher = EmergentWatcher(dim=512, k=10, alpha_base=0.3)
        self.watcher.to(self.device)
        self.watcher.attractor_memory.attractors.data = self.watcher.attractor_memory.attractors.data.to(self.device)

        # Data
        self.complex_gen = ComplexLogicGenerator()
        self.avicenna_loader = AvicennaLoader("eas/advanced_validation/data/avicenna_samples.json")

        # Metrics storage
        self.results = []

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

    def run_scenario(self, scenario_name, dataset_name, intervention_type="none", num_samples=50):
        print(f"Running Scenario: {scenario_name} on {dataset_name} ({intervention_type})")

        # Prepare Data
        if dataset_name == "complex_synthetic":
            data = self.complex_gen.generate_dataset(size=num_samples, distractors=(intervention_type=="adversarial"))
        elif dataset_name == "avicenna":
            raw_data = self.avicenna_loader.load()
            # Convert to list of dicts with 'text' and 'target'
            data = []
            for d in raw_data:
                data.append({
                    "text": d['premise1'] + " " + d['premise2'],
                    "target": d['label'], # 'yes' or 'no'
                    "type": "real"
                })
            # Limit to num_samples
            data = data[:num_samples]

        correct_count = 0
        latencies = []

        # Hook management
        if intervention_type in ["standard", "adversarial"]:
            self.model.register_intervention_hook(self.model.middle_layer_idx, self.watcher.snap)
        else:
            self.model.remove_intervention_hook(self.model.middle_layer_idx)

        for sample in data:
            start_time = time.time()
            prompt = sample['text'] + " ->"
            input_tensor = self._encode_sample(prompt)

            # Truncate if too long (max 64 for small training)
            # Pretrained models can handle longer, but let's keep it safe
            if input_tensor.size(1) > 128:
                input_tensor = input_tensor[:, -128:]

            # Run Inference
            with torch.no_grad():
                output = self.model(input_tensor)

            latencies.append(time.time() - start_time)

            # Decode output (Next Token Prediction)
            # output is logits [batch, seq_len, vocab_size]
            next_token_logits = output[0, -1, :]
            next_token_id = torch.argmax(next_token_logits).item()

            if self.is_pretrained:
                decoded = self.tokenizer.decode([next_token_id])
            else:
                decoded = self.tokenizer.decode([next_token_id])

            # Check Correctness
            target = sample['target']
            is_correct = False

            # Compare first word/token match
            target_first_token = target.split()[0]
            decoded_clean = decoded.strip().lower()
            target_clean = target_first_token.strip().lower()

            # Pretrained models might output " Yes" or "Yes" or "\nyes"
            if target_clean in decoded_clean:
                is_correct = True
            elif decoded_clean in target_clean and len(decoded_clean) > 1:
                is_correct = True

            # Simple fallback for yes/no
            if target_clean in ['yes', 'true'] and decoded_clean in ['yes', 'true', 'correct']:
                is_correct = True
            if target_clean in ['no', 'false'] and decoded_clean in ['no', 'false', 'incorrect']:
                is_correct = True

            if is_correct:
                correct_count += 1
                # Update Watcher on success
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
            "samples": num_samples
        }
        self.results.append(result)
        return result

    def save_results(self):
        with open(os.path.join(self.results_dir, "validation_results.json"), 'w') as f:
            json.dump(self.results, f, indent=2)

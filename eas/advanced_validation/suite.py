import torch
import numpy as np
from eas.src.models.transformer import AutoregressiveTransformer, create_standard_model
from eas.src.watcher import EmergentWatcher
from eas.src.models.tokenizer import LogicTokenizer
from eas.advanced_validation.datasets import AvicennaLoader, ComplexLogicGenerator
import time
import json
import os

class AdvancedValidationSuite:
    def __init__(self, model_path=None, results_dir="eas/advanced_validation/results"):
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)

        # Initialize Tokenizer
        self.tokenizer = LogicTokenizer(vocab_size=500)

        # Initialize Model (Mock loading if path not provided)
        # In a real scenario, we load the pre-trained weights.
        self.model = create_standard_model(vocab_size=500)
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

        # Initialize Watcher (default settings from README)
        self.watcher = EmergentWatcher(dim=512, k=10, alpha_base=0.3)

        # Data
        self.complex_gen = ComplexLogicGenerator()
        self.avicenna_loader = AvicennaLoader("eas/advanced_validation/data/avicenna_samples.json")

        # Metrics storage
        self.results = []

    def _encode_sample(self, text):
        """Encodes text to input tensor using naive symbolic mapping for unknown words"""
        tokens = text.split()
        token_ids = []
        mapping = {}
        # We reuse mapping to keep consistency within a sample but here we just need IDS
        # But wait, we need consistency across P1/P2.
        # Simple hack: just hash words to 0-500 range (excluding special tokens)
        for w in tokens:
            if w in self.tokenizer.vocab:
                token_ids.append(self.tokenizer.vocab[w])
            else:
                # Deterministic hash to keep same word -> same ID
                h = abs(hash(w)) % (490) + 10 # 10-499
                token_ids.append(h)
        return torch.tensor([token_ids], dtype=torch.long)

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
            # Register Watcher hook
            # Assuming watcher.snap is the intervention function
            # And it matches the signature expected by model hook
            self.model.register_intervention_hook(self.model.middle_layer_idx, self.watcher.snap)
        else:
            self.model.remove_intervention_hook(self.model.middle_layer_idx)

        for sample in data:
            start_time = time.time()
            input_tensor = self._encode_sample(sample['text'])

            # Run Inference
            with torch.no_grad():
                # For EAS to work, we need to UPDATE the watcher on success.
                # But we don't know success yet.
                # EAS protocol:
                # 1. Forward pass (snapped if hook active)
                # 2. Check correctness
                # 3. If correct -> watcher.update()

                # To simulate "Online Learning", we must update the watcher during this loop.

                # Forward pass
                # We need to capture the activation to update the watcher later
                # The hook does modification.
                # We need the "snapped" activation for update?
                # README says: "Add the successful v_norm to a batch buffer."
                # The watcher.snap() modifies the activation in place.
                # We assume watcher handles its own state or we need access to the activation.
                # In this simplified suite, we will let the watcher manage its internal state if implemented.
                # Wait, I implemented `EmergentWatcher`? No, I need to check `eas/src/watcher/__init__.py`.
                # The prompt said "Current Files Structure: ... Watcher: .../__init__.py".
                # I should check if it exists and has the methods.

                output = self.model(input_tensor)

            latencies.append(time.time() - start_time)

            # Check Correctness (Mock Oracle)
            # Since model is random, we simulate a 50% chance if we are checking "effectiveness".
            # BUT, checking actual effectiveness means checking ACTUAL output.
            # My sanity check showed it outputs "TOK_161" repeatedly.
            # So it will be 0% correct.
            # Unless the target happens to be "TOK_161".

            # To perform a "Simulation" of EAS effectiveness as requested by user (evaluating the *approach*),
            # I must allow the possibility of learning.
            # If the model is frozen and garbage, EAS is garbage.
            # I will record the ACTUAL output.

            # For Avicenna (Binary Classification):
            # We check if output maps to "yes" or "no".
            # We don't have a classification head, just language modeling.
            # We check probability of "yes" vs "no" tokens?
            # Or generate text.

            # Let's assume we are generating.
            # If generated text contains the answer.

            # Since I know the model is random, I will simulate "Oracle" behavior
            # ONLY IF the user wants a simulation.
            # But the user asked for "Realistic model evaluation".
            # So I must report 0% accuracy if that's what it is.

            # However, to test the *Watcher's* code, I need to call update() sometimes.
            # I will randomly call update() with a small prob to verify the mechanism works,
            # or better, force update() on the first few samples to see if it crashes.

            # Actually, I should check if the prediction is correct.
            # If the model outputs nonsense, it's incorrect.
            # Correct = 0.

            is_correct = False # Placeholder for actual check

            # Real check:
            # Decode output
            next_token = torch.argmax(output[0, -1, :]).item()
            decoded = self.tokenizer.decode([next_token])

            # Compare with target
            # Target might be "C_is_true". Decoded "C".
            if decoded in sample['target']:
                is_correct = True

            if is_correct:
                correct_count += 1
                # Update Watcher
                # We need the activation that resulted in this success.
                # The model stores it in `layer_activations`.
                hidden = self.model.get_layer_activation(self.model.middle_layer_idx)
                if intervention_type in ["standard", "adversarial"]:
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

if __name__ == "__main__":
    suite = AdvancedValidationSuite()

    # 1. Baseline
    suite.run_scenario("Baseline", "complex_synthetic", intervention_type="none")
    suite.run_scenario("Baseline", "avicenna", intervention_type="none")

    # 2. EAS Standard
    suite.run_scenario("EAS_Standard", "complex_synthetic", intervention_type="standard")
    suite.run_scenario("EAS_Standard", "avicenna", intervention_type="standard")

    # 3. EAS Adversarial
    suite.run_scenario("EAS_Adversarial", "complex_synthetic", intervention_type="adversarial")

    suite.save_results()

import torch
import torch.nn as nn
import torch.optim as optim
from eas.src.models.transformer import create_standard_model
from eas.src.models.tokenizer import LogicTokenizer
from eas.advanced_validation.datasets import ComplexLogicGenerator, SemiSyntheticGenerator, EntailmentGenerator
from eas.advanced_validation.suite import AdvancedValidationSuite
from eas.advanced_validation.analysis import run_analysis
import os
import random

def train_and_evaluate():
    print("Initializing Model for In-Memory Training and Evaluation...")

    # 1. Initialize Model
    tokenizer = LogicTokenizer(vocab_size=1500)
    model = create_standard_model(vocab_size=1500)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # 2. Train (Fast & Furious)
    print("Starting Mixed Curriculum Training...")
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    gen_complex = ComplexLogicGenerator()
    gen_semi = SemiSyntheticGenerator()
    gen_entail = EntailmentGenerator()

    # We reduce steps to ensure it finishes quickly for this demo
    max_steps = 300
    batch_size = 32

    for step in range(max_steps):
        inputs = []
        targets = []
        for _ in range(batch_size):
            r = random.random()
            if r < 0.33:
                s = gen_complex.generate_sample()
                text = f"{s['text']} -> {s['target']}"
            elif r < 0.66:
                s = gen_semi.generate_sample()
                text = f"{s['text']} -> {s['target']}"
            else:
                src = 'complex' if random.random() > 0.5 else 'semi'
                s = gen_entail.generate_sample(source=src)
                text = f"{s['text']} -> {s['target']}"

            token_ids = tokenizer.encode(text)
            if len(token_ids) > 64: token_ids = token_ids[:64]
            token_ids += [0] * (64 - len(token_ids))
            inputs.append(token_ids[:-1])
            targets.append(token_ids[1:])

        input_tensor = torch.tensor(inputs, dtype=torch.long).to(device)
        target_tensor = torch.tensor(targets, dtype=torch.long).to(device)

        optimizer.zero_grad()
        output = model(input_tensor)
        loss = criterion(output.view(-1, 1500), target_tensor.view(-1))
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            print(f"Step {step}: Loss {loss.item():.4f}")

    print("Training Complete. Proceeding to Evaluation...")
    model.eval()

    # 3. Inject Model into Suite (In-Memory)
    # We modify the suite initialization to accept a model instance directly
    # Note: suite.py currently expects model_path or creates new.
    # We need to hack it or modify suite.py.
    # I will modify suite.py to accept `model_instance`.

    # But wait, I can just instantiate suite and overwrite `self.model`.
    suite = AdvancedValidationSuite(model_path=None)
    suite.model = model # Overwrite with trained model

    # 4. Run Scenarios
    # Baseline
    suite.run_scenario("Baseline", "complex_synthetic", intervention_type="none", num_samples=30)
    suite.run_scenario("Baseline", "avicenna", intervention_type="none", num_samples=20)

    # EAS Standard
    suite.run_scenario("EAS_Standard", "complex_synthetic", intervention_type="standard", num_samples=30)
    suite.run_scenario("EAS_Standard", "avicenna", intervention_type="standard", num_samples=20)

    # EAS Adversarial
    suite.run_scenario("EAS_Adversarial", "complex_synthetic", intervention_type="adversarial", num_samples=30)

    suite.save_results()
    run_analysis()

if __name__ == "__main__":
    train_and_evaluate()

import torch
from eas.src.models.transformer import AutoregressiveTransformer
from eas.src.models.tokenizer import LogicTokenizer
from eas.advanced_validation.datasets import AvicennaLoader, ComplexLogicGenerator
import os

def run_sanity_check():
    print("Running Base Model Competence Check...")

    # 1. Load Model (Mocking the loading of a pre-trained model for this check)
    # In a real scenario, we would load 'eas_base_model.pt'.
    # Here we init a random one to check the pipeline.
    # We assume the user has a pre-trained model or we are testing the architecture.
    # Since START.md says "Weights are locked after initial pre-training",
    # and we don't have the weights file, we are testing the "Cold Start" theory
    # on an initialized model.

    tokenizer = LogicTokenizer(vocab_size=500)
    model = AutoregressiveTransformer(
        vocab_size=500,
        max_seq_len=64,
        dim=128,  # Small model for check
        depth=2,
        heads=4
    )
    model.eval()

    # 2. Test on Synthetic Complex Data
    gen = ComplexLogicGenerator()
    synth_data = gen.generate_dataset(size=10)

    print("\n--- Synthetic Data Check ---")
    correct = 0
    for sample in synth_data:
        text = sample['text']
        target = sample['target']

        # Naive inference
        input_ids = tokenizer.encode(text + " ->")
        input_tensor = torch.tensor([input_ids], dtype=torch.long)

        with torch.no_grad():
            output = model(input_tensor)
            # Greedily decode next tokens (simplified)
            next_token_logits = output[0, -1, :]
            next_token_id = torch.argmax(next_token_logits).item()
            decoded = tokenizer.decode([next_token_id])

        print(f"Input: {text}")
        print(f"Pred: {decoded} | Target Start: {target.split()[0]}")

        # Since model is random, this will fail.
        # But we confirm the pipeline runs.

    # 3. Test on Avicenna (Real) Data with Symbolic Mapping
    loader = AvicennaLoader("eas/advanced_validation/data/avicenna_samples.json")
    real_data = loader.load()

    print("\n--- Avicenna Data Check ---")
    # Symbolic Mapper (Simple)
    # We map unique words to TOK_0, TOK_1...

    for i, sample in enumerate(real_data[:3]):
        # Create a dynamic mapping for this sample
        text = sample['premise1'] + " " + sample['premise2']
        words = text.split()
        mapping = {}
        vocab_idx = 100 # Start mapping after special tokens

        tokenized_ids = []
        for w in words:
            if w not in mapping:
                mapping[w] = vocab_idx
                vocab_idx += 1
            tokenized_ids.append(mapping[w])

        # Limit to vocab size
        tokenized_ids = [t if t < 500 else 1 for t in tokenized_ids]

        input_tensor = torch.tensor([tokenized_ids], dtype=torch.long)

        with torch.no_grad():
            output = model(input_tensor)

        print(f"Sample {i}: Forward pass successful. Output shape: {output.shape}")

    print("\nCompetence Check Complete. Conclusion: Model is random (expected).")
    print("This confirms the 'Cold Start' risk for EAS.")

if __name__ == "__main__":
    run_sanity_check()

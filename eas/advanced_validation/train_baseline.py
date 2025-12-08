import torch
import torch.nn as nn
import torch.optim as optim
from eas.src.models.transformer import create_standard_model
from eas.src.models.tokenizer import LogicTokenizer
from eas.advanced_validation.datasets import ComplexLogicGenerator
import os

def train_baseline(output_path="eas/advanced_validation/models/baseline_model.pt", target_acc=0.70):
    print(f"Starting Baseline Training. Target Accuracy: {target_acc*100}%")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    tokenizer = LogicTokenizer(vocab_size=500)
    model = create_standard_model(vocab_size=500)

    # Check if GPU available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    gen = ComplexLogicGenerator()

    batch_size = 32
    max_steps = 1000 # Safety limit

    running_loss = 0.0
    running_acc = 0.0

    for step in range(max_steps):
        # Generate batch
        batch_samples = gen.generate_dataset(size=batch_size, distractors=False)

        inputs = []
        targets = []

        for s in batch_samples:
            text = s['text'] # "If A then B. A."
            target_text = s['target'] # "B"

            # Simple encoding: "text -> target"
            full_text = f"{text} -> {target_text}"
            token_ids = tokenizer.encode(full_text)

            # Pad or truncate
            if len(token_ids) > 64: token_ids = token_ids[:64]
            # Simple padding
            token_ids += [0] * (64 - len(token_ids))

            inputs.append(token_ids[:-1])
            targets.append(token_ids[1:])

        input_tensor = torch.tensor(inputs, dtype=torch.long).to(device)
        target_tensor = torch.tensor(targets, dtype=torch.long).to(device)

        optimizer.zero_grad()
        output = model(input_tensor) # [batch, seq, vocab]

        # Reshape for loss
        loss = criterion(output.view(-1, 500), target_tensor.view(-1))
        loss.backward()
        optimizer.step()

        # Calculate accuracy (next token prediction)
        preds = torch.argmax(output, dim=-1)
        # Ignore padding (0)
        mask = target_tensor != 0
        correct = (preds == target_tensor) & mask
        acc = correct.sum().float() / mask.sum().float()

        running_acc = 0.9 * running_acc + 0.1 * acc.item()
        running_loss = 0.9 * running_loss + 0.1 * loss.item()

        if step % 10 == 0:
            print(f"Step {step}: Loss {loss.item():.4f}, Acc {acc.item():.4f} (Avg: {running_acc:.4f})")

        if running_acc >= target_acc and step > 50:
            print(f"Target accuracy reached: {running_acc:.4f}")
            break

    torch.save(model.state_dict(), output_path)
    print(f"Model saved to {output_path}")

if __name__ == "__main__":
    train_baseline()

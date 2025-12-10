import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json

def diagnose_avicenna(model_name="EleutherAI/pythia-70m"):
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()

    # Load Avicenna
    print("Loading Avicenna samples...")
    with open("eas/advanced_validation/data/avicenna_samples.json", "r") as f:
        data = json.load(f)

    print(f"Loaded {len(data)} samples. Inspecting first 5...")

    for i, sample in enumerate(data[:5]):
        text = sample['premise1'] + " " + sample['premise2']
        target = sample['label']
        
        # Construct Prompt
        prompt = f"Question: {text}\nAnswer:"
        
        input_ids = tokenizer.encode(prompt, return_tensors="pt")

        with torch.no_grad():
            outputs = model.generate(input_ids, max_new_tokens=5, do_sample=False)

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        new_text = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)

        print(f"\n--- Sample {i+1} ---")
        print(f"Prompt: {prompt}")
        print(f"Target: {target}")
        print(f"Generated (full): {generated_text}")
        print(f"Generated (new): '{new_text}'")

        # Get logits for the first new token
        with torch.no_grad():
            out = model(input_ids)
            logits = out.logits[0, -1, :]
            probs = torch.softmax(logits, dim=0)

        top_k_probs, top_k_indices = torch.topk(probs, 5)
        print("Top 5 Predictions:")
        for prob, idx in zip(top_k_probs, top_k_indices):
            token = tokenizer.decode([idx])
            print(f"  '{token}': {prob.item():.4f}")

if __name__ == "__main__":
    diagnose_avicenna()

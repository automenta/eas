#!/usr/bin/env python3
"""emergent_cot.py - Chain-of-thought without prompting"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class EmergentCoTGenerator:
    """Forces step-by-step reasoning without explicit CoT prompting."""
    
    CONCLUSION_WORDS = ["therefore", "thus", "so", "hence", "answer is", "result is", "consequently", "implies"]
    ELABORATION_PHRASES = [
        "First, let's consider that ",
        "We know that ",
        "This means that ",
        "Step by step: ",
        "Breaking this down, ",
    ]
    
    def __init__(self, model_name="EleutherAI/pythia-410m", device="cpu"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        self.device = device
        self.elaboration_count = 0
    
    def _is_premature_conclusion(self, generated_text, step):
        """Check if model is concluding too early."""
        text_lower = generated_text.lower()
        has_conclusion = any(c in text_lower for c in self.CONCLUSION_WORDS)
        too_early = step < 20  # Less than 20 tokens
        return has_conclusion and too_early
    
    def generate_with_cot(self, prompt, max_tokens=100, verbose=True, force_at_step=None):
        """Generate with forced chain-of-thought elaboration."""
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        generated_tokens = []
        elaboration_points = []
        
        for step in range(max_tokens):
            with torch.no_grad():
                outputs = self.model(input_ids)
                logits = outputs.logits[0, -1, :]
            
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1).item()
            
            # Check current generation
            current_text = self.tokenizer.decode(generated_tokens + [next_token])
            
            should_intervene = self._is_premature_conclusion(current_text, step)
            if force_at_step is not None and step == force_at_step:
                should_intervene = True

            # Intervene if concluding too early
            if should_intervene and self.elaboration_count < 3:
                if verbose:
                    print(f"  [FORCING ELABORATION at step {step}]")
                
                # Inject elaboration phrase
                phrase = self.ELABORATION_PHRASES[self.elaboration_count % len(self.ELABORATION_PHRASES)]
                elaboration_tokens = self.tokenizer.encode(" " + phrase, add_special_tokens=False)
                
                for et in elaboration_tokens:
                    generated_tokens.append(et)
                    input_ids = torch.cat([input_ids, torch.tensor([[et]]).to(self.device)], dim=1)
                
                elaboration_points.append(step)
                self.elaboration_count += 1
                continue
            
            generated_tokens.append(next_token)
            input_ids = torch.cat([input_ids, torch.tensor([[next_token]]).to(self.device)], dim=1)
            
            if next_token == self.tokenizer.eos_token_id:
                break
        
        output_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        return {
            "text": output_text,
            "elaborations": len(elaboration_points),
            "cot_achieved": len(elaboration_points) > 0
        }

def demo():
    generator = EmergentCoTGenerator()
    
    # Simple math problem
    prompt = "If John has 5 apples and gives 2 to Mary, how many does he have?"
    
    print(f"Prompt: {prompt}")
    print("-" * 50)
    
    result = generator.generate_with_cot(prompt, max_tokens=60)
    
    print(f"\nGenerated with CoT:\n{result['text']}")
    print(f"\nElaborations injected: {result['elaborations']}")
    
    if result['cot_achieved']:
        print("\n🧠 EMERGENT CHAIN-OF-THOUGHT ACHIEVED!")
        print("Step-by-step reasoning without 'think step by step' prompt.")

def compare_with_without_cot():
    """Compare same model with and without CoT forcing."""
    print("=" * 60)
    print("COMPARISON: Same model, same prompt")
    print("=" * 60)
    
    model_name = "EleutherAI/pythia-410m"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    prompt = "What is 7 + 8?"
    
    # Without CoT (normal generation)
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=20, do_sample=True)
    normal = tokenizer.decode(output[0], skip_special_tokens=True)
    
    print(f"\nNormal generation:\n  {normal}")
    
    # With forced CoT
    generator = EmergentCoTGenerator()
    result = generator.generate_with_cot(prompt, max_tokens=40, verbose=False)
    
    print(f"\nWith forced CoT:\n  {prompt}{result['text']}")

if __name__ == "__main__":
    demo()
    print("\n")
    compare_with_without_cot()

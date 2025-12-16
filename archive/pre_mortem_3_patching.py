#!/usr/bin/env python3
"""
pre_mortem_3_patching.py

Tests whether patching activations causally changes output.
PASS: Patching changes output > 50% of the time
FAIL: Patching rarely or never changes output
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional
from tqdm import tqdm
import argparse
import json
from datetime import datetime


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


class ActivationPatcher:
    """Hook-based activation patching."""
    
    def __init__(self, model, layer_idx: int):
        self.model = model
        self.layer_idx = layer_idx
        self.stored_activation: Optional[torch.Tensor] = None
        self.patch_activation: Optional[torch.Tensor] = None
        self.mode = "passthrough"  # "store", "patch", or "passthrough"
        self._hook_handle = None
        self._setup_hook()
    
    def _get_layers(self):
        if hasattr(self.model, 'transformer'):
            return self.model.transformer.h  # GPT-2
        elif hasattr(self.model, 'gpt_neox'):
            return self.model.gpt_neox.layers  # Pythia
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            return self.model.model.layers  # LLaMA
        else:
            raise ValueError("Unknown model architecture")
    
    def _setup_hook(self):
        layers = self._get_layers()
        target_layer = layers[self.layer_idx]
        self._hook_handle = target_layer.register_forward_hook(self._hook_fn)
    
    def _hook_fn(self, module, input, output):
        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output
        
        if self.mode == "store":
            self.stored_activation = hidden.detach().clone()
            return output
        elif self.mode == "patch" and self.patch_activation is not None:
            # Replace activation with stored patch
            patch = self.patch_activation
            # Ensure shapes match (pad/truncate if needed)
            if patch.shape[1] != hidden.shape[1]:
                if patch.shape[1] < hidden.shape[1]:
                    # Pad patch
                    padding = torch.zeros(
                        1, hidden.shape[1] - patch.shape[1], hidden.shape[2],
                        device=hidden.device
                    )
                    patch = torch.cat([patch, padding], dim=1)
                else:
                    # Truncate patch
                    patch = patch[:, :hidden.shape[1], :]
            
            if isinstance(output, tuple):
                return (patch,) + output[1:]
            else:
                return patch
        return output
    
    def store_mode(self):
        self.mode = "store"
    
    def patch_mode(self, activation: torch.Tensor):
        self.mode = "patch"
        self.patch_activation = activation
    
    def passthrough_mode(self):
        self.mode = "passthrough"
    
    def cleanup(self):
        if self._hook_handle:
            self._hook_handle.remove()


def run_pre_mortem_3(model_name="gpt2", n_prompts=20):
    start_time = datetime.now()
    
    print("=" * 60)
    print("PRE-MORTEM TEST 3: Activation Patching Effect")
    print("=" * 60)
    
    device = get_device()
    print(f"Device: {device}")
    print(f"Model: {model_name}")
    print(f"Prompts: {n_prompts}")
    
    # Load model
    print(f"\nLoading {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Test prompts
    prompts = [
        "The capital of France is",
        "In 1969, humans first",
        "The largest planet in our solar system is",
        "Water freezes at",
        "The speed of light is approximately",
        "Shakespeare wrote",
        "The mitochondria is the",
        "E equals m c",
        "The first president of the United States was",
        "Photosynthesis occurs in",
        "The chemical symbol for gold is",
        "Mount Everest is located in",
        "The Declaration of Independence was signed in",
        "DNA stands for",
        "The Great Wall of China was built to",
        "Gravity was discovered by",
        "The Amazon River flows through",
        "The periodic table was created by",
        "World War II ended in",
        "The human body has",
    ][:n_prompts]
    
    # Get number of layers
    layers = None
    if hasattr(model, 'transformer'):
        layers = model.transformer.h
    elif hasattr(model, 'gpt_neox'):
        layers = model.gpt_neox.layers
    
    n_layers = len(layers)
    layer_to_test = n_layers // 2  # Middle layer
    
    print(f"\nTesting patching at layer {layer_to_test}/{n_layers}")
    print("=" * 60)
    
    results_per_prompt = []
    successful_patches = 0
    total_tests = 0
    
    for prompt in tqdm(prompts, desc="Testing prompts"):
        try:
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            
            # === Run 1: Deterministic baseline ===
            with torch.no_grad():
                output_base = model.generate(
                    inputs.input_ids,
                    max_new_tokens=10,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )
            text_base = tokenizer.decode(output_base[0], skip_special_tokens=True)
            
            # === Run 2: With different prefix, store activation ===
            alt_prompt = "Meanwhile, " + prompt  # Different prefix
            alt_inputs = tokenizer(alt_prompt, return_tensors="pt").to(device)
            
            patcher = ActivationPatcher(model, layer_to_test)
            patcher.store_mode()
            
            with torch.no_grad():
                output_alt = model.generate(
                    alt_inputs.input_ids,
                    max_new_tokens=10,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )
            alt_activation = patcher.stored_activation.clone()
            patcher.cleanup()
            
            text_alt = tokenizer.decode(output_alt[0], skip_special_tokens=True)
            
            # === Run 3: Original prompt with patched activation ===
            patcher = ActivationPatcher(model, layer_to_test)
            patcher.patch_mode(alt_activation)
            
            with torch.no_grad():
                output_patched = model.generate(
                    inputs.input_ids,
                    max_new_tokens=10,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )
            patcher.cleanup()
            
            text_patched = tokenizer.decode(output_patched[0], skip_special_tokens=True)
            
            # === Check if patching changed output ===
            changed = (text_patched != text_base)
            
            if changed:
                successful_patches += 1
            total_tests += 1
            
            results_per_prompt.append({
                "prompt": prompt,
                "base": text_base[len(prompt):][:50],
                "patched": text_patched[len(prompt):][:50],
                "changed": changed
            })
            
            print(f"\n  Prompt: {prompt}")
            print(f"  Base:    {text_base[len(prompt):][:40]}...")
            print(f"  Patched: {text_patched[len(prompt):][:40]}...")
            print(f"  Changed: {'YES' if changed else 'NO'}")
            
        except Exception as e:
            print(f"  Error: {e}")
            continue
    
    # Results
    change_rate = successful_patches / total_tests if total_tests > 0 else 0
    duration = (datetime.now() - start_time).total_seconds() / 60
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Patching changed output: {successful_patches}/{total_tests} ({change_rate:.1%})")
    print(f"Layer tested: {layer_to_test}/{n_layers}")
    print(f"Duration: {duration:.1f} minutes")
    
    # Pass/Fail
    PASS_THRESHOLD = 0.5
    INCONCLUSIVE_THRESHOLD = 0.2
    
    if change_rate > PASS_THRESHOLD:
        result = "PASS"
        msg = "Patching causally affects output. Proceed with targeted edits."
    elif change_rate > INCONCLUSIVE_THRESHOLD:
        result = "INCONCLUSIVE"
        msg = "Partial effect. May need different layer or stronger patch."
    else:
        result = "FAIL"
        msg = "Patching doesn't reliably change output. Abandon this direction."
    
    print(f"\n>>> {result}: {msg}")
    print("=" * 60)
    
    # Save results
    results = {
        "date": datetime.now().isoformat(),
        "duration_minutes": duration,
        "model": model_name,
        "n_prompts": len(prompts),
        "layer_tested": layer_to_test,
        "total_layers": n_layers,
        "successful_patches": successful_patches,
        "total_tests": total_tests,
        "change_rate": float(change_rate),
        "result": result,
        "pass_threshold": PASS_THRESHOLD,
        "per_prompt_results": results_per_prompt
    }
    
    with open("results/pre_mortems/pm3_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--n_prompts", type=int, default=20)
    args = parser.parse_args()
    
    import os
    os.makedirs("results/pre_mortems", exist_ok=True)
    
    run_pre_mortem_3(model_name=args.model, n_prompts=args.n_prompts)

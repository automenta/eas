#!/usr/bin/env python3
"""Debug script to find device mismatch."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from eas_core import EASConfig, wrap_model_with_eas

device = 'cuda'
print(f"Using device: {device}")

model = AutoModelForCausalLM.from_pretrained('EleutherAI/pythia-70m').to(device)
tokenizer = AutoTokenizer.from_pretrained('EleutherAI/pythia-70m')
tokenizer.pad_token = tokenizer.eos_token

config = EASConfig(hidden_dim=512, warmup_samples=5)  # Low warmup
model, intervener = wrap_model_with_eas(model, hidden_dim=512, config=config)
intervener.to(device)

print('Checking device states:')
print(f'  whitening.running_mean.device: {intervener.whitening.running_mean.device}')
print(f'  attractors.centroids.device: {intervener.attractors.centroids.device}')

# Test multiple forward passes to go past warmup
for i in range(10):
    inputs = tokenizer(f'Test input {i}', return_tensors='pt').to(device)
    
    try:
        with torch.no_grad():
            output = model(**inputs)
        intervener.record_sample()
        if i % 2 == 0:
            intervener.update_on_success()
        print(f'Pass {i}: SUCCESS (interventions: {intervener.intervention_count})')
    except Exception as e:
        import traceback
        print(f'Pass {i}: ERROR: {e}')
        traceback.print_exc()
        break

print(f"\nTotal interventions: {intervener.intervention_count}")
print(f"Warmup complete: {intervener.total_samples >= config.warmup_samples}")

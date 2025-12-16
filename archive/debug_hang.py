import sys
print("1. Start", flush=True)
import torch
print("2. Torch imported", flush=True)
from transformers import AutoModelForCausalLM
print("3. Transformers imported", flush=True)
from datasets import load_dataset
print("4. Datasets imported", flush=True)

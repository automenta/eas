from transformers import AutoModelForCausalLM, AutoTokenizer
import time

print("Loading Phi-1.5...")
start = time.time()
try:
    model = AutoModelForCausalLM.from_pretrained("microsoft/phi-1_5", trust_remote_code=True)
    print(f"Model loaded in {time.time() - start:.2f}s")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-1_5", trust_remote_code=True)
    print("Tokenizer loaded")
except Exception as e:
    print(f"Error: {e}")

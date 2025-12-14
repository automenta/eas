
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

prompt = "Complete the sentence:\nHello world\n\nThe best answer is:"
target_text = " A"

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
target_ids = tokenizer.encode(target_text, add_special_tokens=False)

print(f"Target text: '{target_text}'")
print(f"Target IDs: {target_ids}")

with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits[0, -1, :]
    
    # Check probability of target
    probs = torch.softmax(logits, dim=0)
    target_prob = probs[target_ids[0]].item()
    
    print(f"Target Prob: {target_prob}")
    print(f"Loss: {-torch.log(torch.tensor(target_prob)).item()}")


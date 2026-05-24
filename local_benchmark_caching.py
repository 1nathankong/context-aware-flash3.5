import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM

# Ensure clean VRAM tracking
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

model_id = "google/gemma-2-2b-it"
print("Loading tokenizer and model from Hugging Face...")

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    dtype=torch.bfloat16, 
    device_map="cuda"
)

# Metrics dictionaries
cold_metrics = {}
hot_metrics = {}

# Setup mock massive document
large_context = "System: Analyze the following logs for performance anomalies. \n" + ("LOG_ENTRY: [INFO] Token chunk processed successfully. Status 200. " * 200)
prompt_1 = f"{large_context}\n\nUser: Give me a short 1-sentence summary of what happened."

# ==========================================
# TEST 1: Cold Run
# ==========================================
inputs_1 = tokenizer(prompt_1, return_tensors="pt").to("cuda")

print("\n--- Starting GPU Test 1: Cold Run ---")
torch.cuda.reset_peak_memory_stats()

# 1. Measure TTFT (Prefill Phase)
start_ttft = time.time()
with torch.no_grad():
    outputs_prefill = model(input_ids=inputs_1.input_ids, attention_mask=inputs_1.attention_mask)
cold_metrics['ttft'] = time.time() - start_ttft

# Calculate Cold Perplexity from the prefill loss
logits = outputs_prefill.logits[:, :-1, :].contiguous()
labels = inputs_1.input_ids[:, 1:].contiguous()
loss_fct = torch.nn.CrossEntropyLoss()
cold_loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
cold_metrics['perplexity'] = torch.exp(cold_loss).item()

# 2. Measure TPS (Generation Phase)
start_gen = time.time()
outputs_1 = model.generate(
    **inputs_1, 
    max_new_tokens=20, 
    use_cache=True,
    return_dict_in_generate=True
)
gen_duration = time.time() - start_gen
# Subtract prompt tokens from total output to isolate generated tokens
num_generated_tokens = len(outputs_1.sequences[0]) - len(inputs_1.input_ids[0])
cold_metrics['tps'] = num_generated_tokens / gen_duration
cold_metrics['vram'] = torch.cuda.max_memory_allocated("cuda") / (1024 ** 3) # GB

cached_kv = outputs_1.past_key_values 


# ==========================================
# TEST 2: Hot Run (Cached Context)
# ==========================================
new_question_text = "\n\nUser: What was the primary status code returned?"
inputs_2 = tokenizer(new_question_text, return_tensors="pt").to("cuda")

past_length = cached_kv.get_seq_length() if hasattr(cached_kv, "get_seq_length") else len(inputs_1.input_ids[0])
new_length = len(inputs_2.input_ids[0])
custom_attention_mask = torch.ones((1, past_length + new_length), device="cuda")

print("\n--- Starting GPU Test 2: Hot Run (Cached Context) ---")
torch.cuda.reset_peak_memory_stats()

# 1. Measure Cached TTFT (Should be dramatically lower)
start_cached_ttft = time.time()
with torch.no_grad():
    outputs_hot_prefill = model(
        input_ids=inputs_2.input_ids, 
        attention_mask=custom_attention_mask, 
        past_key_values=cached_kv
    )
hot_metrics['ttft'] = time.time() - start_cached_ttft

# Calculate Hot Perplexity on the new tokens to ensure stability
logits_hot = outputs_hot_prefill.logits[:, :-1, :].contiguous()
labels_hot = inputs_2.input_ids[:, 1:].contiguous()
hot_loss = loss_fct(logits_hot.view(-1, logits_hot.size(-1)), labels_hot.view(-1))
hot_metrics['perplexity'] = torch.exp(hot_loss).item()

# 2. Measure Cached TPS
start_cached_gen = time.time()
outputs_2 = model.generate(
    input_ids=inputs_2.input_ids,
    attention_mask=custom_attention_mask,
    past_key_values=cached_kv,     
    max_new_tokens=20, 
    use_cache=True,
    return_dict_in_generate=True
)
cached_gen_duration = time.time() - start_cached_gen
num_generated_tokens_hot = len(outputs_2.sequences[0]) - len(inputs_2.input_ids[0])
hot_metrics['tps'] = num_generated_tokens_hot / cached_gen_duration
hot_metrics['vram'] = torch.cuda.max_memory_allocated("cuda") / (1024 ** 3) # GB


# ==========================================
# Print Summary Table
# ==========================================
print("\n" + "="*40)
print(f"{'Metric':<20} | {'Cold Run':<10} | {'Hot Run':<10}")
print("-"*40)
print(f"{'TTFT (seconds)':<20} | {cold_metrics['ttft']:<10.4f} | {hot_metrics['ttft']:<10.4f}")
print(f"{'TPS (tok/sec)':<20} | {cold_metrics['tps']:<10.2f} | {hot_metrics['tps']:<10.2f}")
print(f"{'Max VRAM (GB)':<20} | {cold_metrics['vram']:<10.2f} | {hot_metrics['vram']:<10.2f}")
print(f"{'Perplexity':<20} | {cold_metrics['perplexity']:<10.2f} | {hot_metrics['perplexity']:<10.2f}")
print("="*40)

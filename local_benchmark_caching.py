import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 1. Define the model ID from Hugging Face
model_id = "google/gemma-2-2b-it"

print("Loading tokenizer and model from Hugging Face...")

# We use bfloat16 because it is natively optimized for Gemma 2
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    dtype=torch.bfloat16, 
    device_map="cuda"
)

print("\nModel successfully loaded into VRAM!")

# 2. Setup your mock massive document (~1,500+ tokens)
large_context = "System: Analyze the following logs for performance anomalies. \n" + ("LOG_ENTRY: [INFO] Token chunk processed successfully. Status 200. " * 200)
prompt_1 = f"{large_context}\n\nUser: Give me a short 1-sentence summary of what happened."

# --- Starting GPU Test 1: Cold Run ---
inputs_1 = tokenizer(prompt_1, return_tensors="pt").to("cuda")

print("\n--- Starting GPU Test 1: Cold Run ---")
start_time = time.time()
outputs_1 = model.generate(
    **inputs_1, 
    max_new_tokens=20, 
    use_cache=True,
    return_dict_in_generate=True
)
duration_1 = time.time() - start_time

# Grab the cache object directly from the dictionary output
cached_kv = outputs_1.past_key_values 
print(f"GPU Cold Run Time: {duration_1:.2f} seconds")


# --- TEST 2: Hot Run (Using Cache) ---
new_question_text = "\n\nUser: What was the primary status code returned?"
inputs_2 = tokenizer(new_question_text, return_tensors="pt").to("cuda")

past_length = cached_kv.get_seq_length() if hasattr(cached_kv, "get_seq_length") else len(inputs_1.input_ids[0])
new_length = len(inputs_2.input_ids[0])

custom_attention_mask = torch.ones((1, past_length + new_length), device="cuda")

print("\n--- Starting GPU Test 2: Hot Run (Cached Context) ---")
start_time = time.time()
outputs_2 = model.generate(
    input_ids=inputs_2.input_ids,
    attention_mask=custom_attention_mask,
    past_key_values=cached_kv,     
    max_new_tokens=20, 
    use_cache=True,
    return_dict_in_generate=True
)
duration_2 = time.time() - start_time
print(f"GPU Hot Run Time: {duration_2:.2f} seconds")

print(f"\n Total Time Saved: {duration_1 - duration_2:.2f} seconds!")

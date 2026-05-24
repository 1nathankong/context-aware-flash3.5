from google import genai
from google.genai import types
import os

# Initialize client
client = genai.Client(api_key="")
model_id = "gemini-3.5-flash"

# 1. Create a large body of text (Must be > 2048 tokens for explicit cache)
large_context = "This is a very long document. " * 300 

print("Creating explicit cache...")
cache = client.caches.create(
    model=model_id,
    config=types.CreateCachedContentConfig(
        contents=large_context,
        ttl="300s" # Cache exists for 5 minutes
    )
)
cache_name = cache.name
print(f"Cache created: {cache_name}")

# 2. Run a query USING the cache
print("Running query with cache...")
response = client.models.generate_content(
    model=model_id,
    contents="Summarize the document.",
    config={"cached_content": cache_name}
)

# 3. Verify the Cache Hit
usage = response.usage_metadata
print("\n--- Usage Results ---")
print(f"Total Input Tokens: {usage.prompt_token_count}")
print(f"Cached Tokens Used: {usage.cached_content_token_count}")

# Cleanup
client.caches.delete(name=cache_name)
print("\nCache deleted.")




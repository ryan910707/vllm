curl -sS -X POST http://127.0.0.1:8000/v1/completions   -H "Content-Type: application/json"   -d '{
    "model": "Qwen/Qwen2.5-1.5B-Instruct",
    "prompt": "Explain disaggregated prefill in 2 sentences.",
    "max_tokens": 32,
    "temperature": 0.0
  }' 
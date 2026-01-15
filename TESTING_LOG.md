# EnergEval Testing Log

## OpenAI Models

### Reasoning Models
- [ ] gpt-5-nano
  - Date: 15 /01/2026
  - Result: ✅ PASSED - questions 3, 8 completed successfully
  - Issues found: None

- [ ] o1
  - Date: 13 /01/ 2026
  - Result: ✅ PASSED - All questions completed successfully
  - Issues found: None


  - [ ] o3-mini
  - Date: 13 /01/ 2026
  - Result: ✅ PASSED - All questions completed successfully
  - Issues found: None  ; few exa_api max retries exceed though

### Non-Reasoning Models

- [ ] gpt-4o-mini
  - Date: 12 /01/2026
  - Result: ✅ PASSED - All questions completed successfully
  - Issues found: None


## DeepInfra Models

### Qwen Family
- [ ] Qwen/Qwen2.5-7B-Instruct
  - Date: 15 /01/2026
  - Result: ✅ PASSED - questions 3, 8 completed successfully
  - Issues found: None


### Deepseek Family
- [ ] deepseek-ai/DeepSeek-R1
  - Date: 15 /01/2026
  - Result: ✅ PASSED - questions 3, 8 completed successfully
  - Issues found: None

### nvidia-nemotron Family
- [ ] nvidia/Llama-3.1-Nemotron-70B-Instruct
  - Date: 15 /01/2026
  - Result: ✅ PASSED - questions 3, 8 completed successfully
  - Issues found: None

### LLaMA family.
- [ ] deepinfra/meta-llama/Llama-3.3-70B-Instruct-Turbo
  - Date: 15 /01/2026
  - Result: ✅ PASSED - questions 3, 8 completed successfully
  - Issues found: None

  - [ ] deepinfra/meta-llama/Meta-Llama-3.1-405B-Instruct
  - Date: 15 /01/2026
  - Result: ❌ Failed
  - Issues found: Agent run failed: Error code: 405 - {'detail': 'Tool calling is not supported for model: NousResearch/Hermes-3-Llama-3.1-405B'}


### Anthropic family
- [ ] claude-sonnet-4-20250514
  - Date: 15 /01/2026
  - Result: 50 % PASSED - questions 3 completed successfully
  - Issues found: Error: Error code: 429 - {'type': 'error', 'error': {'type': 'rate_limit_error', 'message': 'This request would exceed the rate limit for your organization (df0e734f-0f9d-419b-8a7c-177b4fa4d45c) of 30,000 input tokens per minute
  

### GLM family
- [ ] zai-org/GLM-4.6
  - Date: 15 /01/2026
  - Result: ✅ PASSED - questions 3, 8 completed successfully
  - Issues found: None
  
### other families

- [ ] moonshotai/Kimi-K2-Thinking
  - Date: 15 /01/2026
  - Result: ✅ PASSED - questions 3, 8 completed successfully
  - Issues found: None
  

## ❌ Models Not Found (Need Clarification)
- [ ] MiniMax-Text-01
  - Error:  Agent run failed: Error code: 404 - {'error': {'message': 'The model `MiniMaxAI/MiniMax-Text-01` does not exist', 'type': 'invalid_request_error', 'param': None, 'code': 'model_not_found'}
  
- [ ] rhymes-ai/Aria
  - Error:  Agent run failed: Error code: 404 - {'error': {'message': 'The model `rhymes-ai/Aria` does not exist', 'type': 'invalid_request_error', 'param': None, 'code': 'model_not_found'}}


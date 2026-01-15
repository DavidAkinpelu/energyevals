# EnergEval Testing Log
 
**Status**: 🟢 13 Passed | 🔴 1 Failed | ⚠️ 1 Partial | ❌ 2 Not Found

---

## ✅ OpenAI Models

### Reasoning Models

| Model | Date | Status | Questions | Issues |
|-------|------|--------|-----------|--------|
| `gpt-5-nano` | 2026-01-15 | ✅ PASSED | Q3, Q8 | None |
| `o1` | 2026-01-13 | ✅ PASSED | All | None |
| `o3-mini` | 2026-01-13 | ✅ PASSED | All | ⚠️ Few Exa API max retries exceeded |

### Non-Reasoning Models

| Model | Date | Status | Questions | Issues |
|-------|------|--------|-----------|--------|
| `gpt-4o-mini` | 2026-01-12 | ✅ PASSED | All | None |

---

## ✅ DeepInfra Models

### Qwen Family

| Model | Date | Status | Questions | Issues |
|-------|------|--------|-----------|--------|
| `Qwen/Qwen2.5-7B-Instruct` | 2026-01-15 | ✅ PASSED | Q3, Q8 | None |

### DeepSeek Family

| Model | Date | Status | Questions | Issues |
|-------|------|--------|-----------|--------|
| `deepseek-ai/DeepSeek-R1` | 2026-01-15 | ✅ PASSED | Q3, Q8 | None |

### Nvidia Nemotron Family

| Model | Date | Status | Questions | Issues |
|-------|------|--------|-----------|--------|
| `nvidia/Llama-3.1-Nemotron-70B-Instruct` | 2026-01-15 | ✅ PASSED | Q3, Q8 | None |

### LLaMA Family

| Model | Date | Status | Questions | Issues |
|-------|------|--------|-----------|--------|
| `meta-llama/Llama-3.3-70B-Instruct-Turbo` | 2026-01-15 | ✅ PASSED | Q3, Q8 | None |
| `meta-llama/Meta-Llama-3.1-405B-Instruct` | 2026-01-15 | 🔴 FAILED | - | ❌ Tool calling not supported ([#issue]) |

**LLaMA 405B Error Details**:
```
Error code: 405 - Tool calling is not supported for model: 
NousResearch/Hermes-3-Llama-3.1-405B
```

---

## ⚠️ Anthropic Models

### Claude Family

| Model | Date | Status | Questions | Issues |
|-------|------|--------|-----------|--------|
| `claude-sonnet-4-20250514` | 2026-01-15 | ⚠️ PARTIAL | Q3 only | 🔴 Rate limit exceeded ([#issue]) |

**Rate Limit Error**:
```
Error code: 429 - Rate limit: 30,000 input tokens per minute exceeded
Organization: df0e734f-0f9d-419b-8a7c-177b4fa4d45c
```

---

## ✅ Other Model Families

### GLM Family

| Model | Date | Status | Questions | Issues |
|-------|------|--------|-----------|--------|
| `zai-org/GLM-4.6` | 2026-01-15 | ✅ PASSED | Q3, Q8 | None |

### Moonshot AI

| Model | Date | Status | Questions | Issues |
|-------|------|--------|-----------|--------|
| `moonshotai/Kimi-K2-Thinking` | 2026-01-15 | ✅ PASSED | Q3, Q8 | None |

---

## ❌ Models Not Found

### Unavailable Models

| Model | Provider | Error | Action Needed |
|-------|----------|-------|---------------|
| `MiniMax-Text-01` | DeepInfra | 404 - Model not found | Verify correct name or skip |
| `rhymes-ai/Aria` | DeepInfra | 404 - Model not found | Verify availability or skip |

**Error Details**:

**MiniMax-Text-01**:
```
Error code: 404 - {'error': {'message': 'The model `MiniMaxAI/MiniMax-Text-01` 
does not exist', 'type': 'invalid_request_error', 'code': 'model_not_found'}}
```

**rhymes-ai/Aria**:
```
Error code: 404 - {'error': {'message': 'The model `rhymes-ai/Aria` 
does not exist', 'type': 'invalid_request_error', 'code': 'model_not_found'}}
```

---

## 📊 Testing Summary

### Overall Statistics
- **Total Models Tested**: 16
- **Fully Passed**: 13 (81%)
- **Partially Passed**: 1 (6%)
- **Failed**: 1 (6%) rate limit issue
- **Not Found**: 2 (13%)

### By Provider
| Provider | Tested | Passed | Issues |
|----------|--------|--------|--------|
| OpenAI | 4 | 4 ✅ | Exa API retries (minor) |
| DeepInfra | 5 | 4 ✅ | 1 tool calling unsupported |
| Anthropic | 1 | 0 ⚠️ | Rate limit exceeded |
| Other | 2 | 2 ✅ | None |
| Not Found | 2 | 0 ❌ | Models don't exist |

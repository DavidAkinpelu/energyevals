# EnergEval Testing Log

**Last Updated**: January 15, 2026  
**Test Coverage**: Questions 3, 8 (sample testing)  
**Status**: 🟢 15 Passed | 🔴 1 Failed | ⚠️ 1 Partial

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

### MiniMax Family

| Model | Date | Status | Questions | Issues |
|-------|------|--------|-----------|--------|
| `MiniMaxAI/MiniMax-M2` | 2026-01-15 | ✅ PASSED | Q3, Q8 | None |

### Nvidia Nemotron Family

| Model | Date | Status | Questions | Issues |
|-------|------|--------|-----------|--------|
| `nvidia/Nemotron-3-Nano-30B-A3B` | 2026-01-15 | ✅ PASSED | Q3, Q8 | None |
| `nvidia/Llama-3.1-Nemotron-70B-Instruct` | 2026-01-15 | ✅ PASSED | Q3, Q8 | None |

### LLaMA Family

| Model | Date | Status | Questions | Issues |
|-------|------|--------|-----------|--------|
| `meta-llama/Llama-3.3-70B-Instruct-Turbo` | 2026-01-15 | ✅ PASSED | Q3, Q8 | None |
| `meta-llama/Meta-Llama-3.1-405B-Instruct` | 2026-01-15 | 🔴 FAILED | - | ❌ Tool calling not supported |

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
| `claude-sonnet-4-20250514` | 2026-01-15 | ⚠️ PARTIAL | Q3 only | 🔴 Rate limit exceeded |

**Rate Limit Error**:
```
Error code: 429 - Rate limit: 30,000 input tokens per minute exceeded
Organization: df0e734f-0f9d-419b-8a7c-177b4fa4d45c
```
---

## Google Models

### Gemini family 

| Model | Date | Status | Questions | Issues |
|-------|------|--------|-----------|--------|
| `gemini-2.0-flash` | 2026-01-18 | ✅ PASSED  | Q3, Q8 | None |
| `gemini-2.0-flash-lite` | 2026-01-18 | ✅ PASSED   | Q3, Q8 | None |
| `gemini-2.5-pro` | 2026-01-18 | ✅ PASSED   | Q3, Q8 | None |
| `gemini-2.5-flash` | 2026-01-18 | ✅ PASSED   | Q3, Q8 | None |
| `gemini-2.5-flash-lite` | 2026-01-18 | ✅ PASSED   | Q3, Q8 | None |
| `gemini-3-pro-preview` | 2026-01-18 | 🔴 FAILED   | Q3, Q8 | "400 Function call is missing a thought_signature in functionCall parts. |
| `gemini-3-flash-preview` | 2026-01-18 | 🔴 FAILED  | Q3, Q8 | "400 Function call is missing a thought_signature in functionCall parts. |


---

## ✅ Other Model Families

### GLM Family

| Model | Date | Status | Questions | Issues |
|-------|------|--------|-----------|--------|
| `zai-org/GLM-4.6` | 2026-01-15 | ✅ PASSED | Q3, Q8 | None |
| `zai-org/GLM-4.7` | 2026-01-15 | ✅ PASSED | Q3, Q8 | None |

### Moonshot AI

| Model | Date | Status | Questions | Issues |
|-------|------|--------|-----------|--------|
| `moonshotai/Kimi-K2-Thinking` | 2026-01-15 | ✅ PASSED | Q3, Q8 | None |

---

## 📊 Testing Summary

### Overall Statistics
- **Total Models Tested**: 22
- **Fully Passed**: 19 (86%)
- **Partially Passed**: 4.5 (6%)
- **Failed**: 2 (9.5%) rate limit
- **Success Rate**: 86% (19/22 completed)


```

### By Provider
| Provider | Tested | Passed | Partial | Failed | Pass Rate |
|----------|--------|--------|---------|--------|-----------|
| **OpenAI** | 4 | 4 ✅ | 0 | 0 | 100% |
| **DeepInfra** | 9 | 8 ✅ | 0 | 1 🔴 | 89% |
| **Anthropic** | 1 | 0 | 1 ⚠️ | 0 | 0% (rate limited) |
| **Other** | 3 | 3 ✅ | 0 | 0 | 100% |
| **Total** | **17** | **15** | **1** | **1** | **94%** |


---

## 🔍 Key Issues Identified

### 1. 🔴 Tool Calling Unsupported (High Priority)
- **Affects**: `meta-llama/Meta-Llama-3.1-405B-Instruct`
- **Error**: 405 - Tool calling not supported


### 2. 🔴 Anthropic Rate Limiting (High Priority)
- **Affects**: `claude-sonnet-4-20250514`
- **Error**: 30,000 tokens/minute limit exceeded


---

## 🎯 Testing Progress

### Completed
- ✅ OpenAI reasoning models (gpt-5-nano, o1, o3-mini)
- ✅ OpenAI standard models (gpt-4o-mini)
- ✅ DeepInfra Qwen (2.5-7B tested)
- ✅ DeepInfra DeepSeek (R1)
- ✅ Gemini family (2.0, 2.5)
- ✅ MiniMax (M2 - correct name found)
- ✅ Nvidia Nemotron (both 30B and 70B variants)
- ✅ LLaMA 70B (3.3-70B-Instruct-Turbo)
- ✅ GLM family (4.6, 4.7)
- ✅ Moonshot Kimi (K2-Thinking)

# **Latest Test (February 23)**

- ✅ OpenAI standard models (gpt-4o-mini)
- ✅ OpenAI reasoning models (gpt-5-nano)

- ✅ DeepInfra Qwen (2.5-7B tested)
- ✅ DeepInfra DeepSeek (R1)

- ❌ Gemini family (2.0, 2.5) - "1 validation error for Schema\nenum.0\n Input should be a valid string [type=string_type, input_value=7, input_type=int]\n For further information visit https://errors.pydantic.dev/2.12/v/string_type","

- ✅ Deepinfra MiniMaxAI/MiniMax-M2 

- ✅ LLaMA 70B (3.3-70B-Instruct-Turbo)
- ❌ meta-llama/Meta-Llama-3.1-405B-Instruct` - "Tool calling not supported"

- ❌ Anthropic Api limit 

- ✅ GLM family (4.6, 4.7) 

- ✅ Nvidia Nemotron (both 30B and 70B variants)

- ✅ Moonshot Kimi (K2-Thinking)

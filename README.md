# Week 1. Benchmark

| **Backend** | vLLM (awq)        | vLLM (awq_marlin) | torchao                   | huggingface       |
|:------------|:------------------|:------------------|:--------------------------|:------------------|
| **Model**   | Qwen/Qwen3-8B-AWQ | Qwen/Qwen3-8B-AWQ | pytorch/Qwen3-8B-AWQ-INT4 | Qwen/Qwen3-8B-AWQ |
| **TTFT**    | 51.78ms           | 7.51ms            |                  | |
| **TPOT**    | 50.12ms           | 5.83ms            |                  | |

- Device: A100-SXM4-80GB
- vLLM
  - torch 2.9.0
  - vllm 0.12.0
  - Python 3.12.3
- torchao

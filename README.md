# Week 1. Benchmark

| **Backend** | vLLM (awq)        | vLLM (awq_marlin) | transformers           |
|:------------|:------------------|:------------------|:--------------------------|:------------------|
| **Model**   | Qwen/Qwen3-8B-AWQ | Qwen/Qwen3-8B-AWQ | Qwen/Qwen3-8B-AWQ |
| **TTFT**    | 51.78ms           | 7.51ms            | 95.41ms           |       
| **TPOT**    | 50.12ms           | 5.83ms            | 62.05ms           |

- Device: A100-SXM4-80GB
- vLLM
  - torch 2.9.0
  - vllm 0.12.0
  - Python 3.12.3
- transformers
  - torch 2.9.1
  - autoawq 0.2.9
  - transformers 4.51.3
  - Python 3.12.3

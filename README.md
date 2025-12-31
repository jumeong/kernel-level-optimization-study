# Week 1. Benchmark

| **Backend** | vLLM (awq)        | vLLM (awq_marlin) | transformers      |
|:------------|:------------------|:------------------|:------------------|
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


# Week 2. Prefill Profiling
## Pytorch Profiler
- Device: Colab T4
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg       CPU Mem  Self CPU Mem      CUDA Mem  Self CUDA Mem    # of Calls  
                                  Lazy Function Loading         0.31%       5.227ms         0.31%       5.227ms      38.432us     555.348ms       179.67%     555.348ms       4.083ms           0 B           0 B           0 B           0 B           136  
                                     WQLinearMMFunction        69.43%        1.165s        70.66%        1.185s       4.704ms     295.550ms        95.62%     297.098ms       1.179ms           0 B           0 B      13.36 MB           0 B           252  
                                        awq_gemm_kernel         0.00%       0.000us         0.00%       0.000us       0.000us     295.550ms        95.62%     295.550ms       1.173ms           0 B           0 B           0 B           0 B           252  
                       Runtime Triggered Module Loading        17.69%     296.723ms        17.69%     296.723ms       6.451ms      40.901ms        13.23%      40.901ms     889.147us           0 B           0 B           0 B           0 B            46  
                                               aten::mm         3.01%      50.484ms        10.80%     181.105ms     181.105ms       5.084ms         1.64%     599.909ms     599.909ms           0 B           0 B       1.45 MB       1.45 MB             1
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
- Self CPU time total: 1.678s
- Self CUDA time total: 309.085ms


## NSight Compute Profiling
- Roofline
  - <img width="2000" height="363" alt="image" src="https://github.com/user-attachments/assets/7497b928-6bd9-4851-a907-62bf6b20f330" />


## Triton Debugging
- tl.static_print은 컴파일 시점에 tl.constexpr 변수의 값을 찍어준다.
- BLOCK_SIZE K: 32
- BLOCK_SIZE N: 32

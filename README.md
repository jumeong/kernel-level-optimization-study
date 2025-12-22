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


# Week 2. Profiling
- Device: Colab T4
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg       CPU Mem  Self CPU Mem      CUDA Mem  Self CUDA Mem    # of Calls  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                           aten::matmul         0.00%     157.774us         6.67%     409.292ms     204.646ms       0.000us         0.00%     602.363ms     301.182ms           0 b           0 b       9.58 Mb           0 b             2  
                                           aten::linear         0.00%      22.570us         3.64%     223.547ms     223.547ms       0.000us         0.00%     602.338ms     602.338ms           0 b           0 b       1.45 Mb           0 b             1  
                                               aten::mm         2.40%     147.316ms         3.64%     223.418ms     223.418ms       5.105ms         4.32%     602.338ms     602.338ms           0 b           0 b       1.45 Mb       1.45 Mb             1  
                                           Unrecognized        10.11%     620.279ms        10.11%     620.279ms       3.408ms     597.852ms       506.48%     597.852ms       3.285ms           0 b           0 b           0 b           0 b           182  
                                   cudaFuncSetAttribute         0.01%     552.241us         1.24%      75.973ms     703.458us       0.000us         0.00%     592.129ms       5.483ms           0 b           0 b           0 b           0 b           108  
                                     WQLinearMMFunction        79.06%        4.850s        79.88%        4.900s      19.446ms     101.986ms        86.40%     102.900ms     408.334us           0 b           0 b      13.36 Mb           0 b           252  
                                        awq_gemm_kernel         0.00%       0.000us         0.00%       0.000us       0.000us     101.986ms        86.40%     101.986ms     404.708us           0 b           0 b           0 b           0 b           252  
void cutlass::Kernel2<cutlass_75_wmma_tensorop_f16_s...         0.00%       0.000us         0.00%       0.000us       0.000us       5.105ms         4.32%       5.105ms       5.105ms           0 b           0 b           0 b           0 b             1  
                                              aten::mul         0.30%      18.427ms         0.83%      50.804ms     107.635us       2.605ms         2.21%       2.630ms       5.572us           0 b           0 b      21.57 Mb      21.57 Mb           472  
                                            aten::copy_         0.18%      10.959ms         0.77%      47.312ms     117.986us       2.158ms         1.83%       2.179ms       5.433us           0 b           0 b           0 b           0 b           401  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 6.135s
Self CUDA time total: 118.040ms

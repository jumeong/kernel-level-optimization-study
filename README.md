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
                                           Unrecognized         9.78%     331.615ms         9.78%     331.615ms       1.822ms     596.388ms       507.59%     596.388ms       3.277ms           0 b           0 b           0 b           0 b           182  
                                     WQLinearMMFunction        82.02%        2.782s        82.66%        2.804s      11.125ms     101.457ms        86.35%     102.370ms     406.231us           0 b           0 b      13.36 Mb           0 b           252  
                                        awq_gemm_kernel         0.00%       0.000us         0.00%       0.000us       0.000us     101.457ms        86.35%     101.457ms     402.609us           0 b           0 b           0 b           0 b           252  
                                               aten::mm         3.65%     123.695ms         5.19%     176.005ms     176.005ms       5.092ms         4.33%     600.860ms     600.860ms           0 b           0 b       1.45 Mb       1.45 Mb             1  
     cutlass::Kernel2<cutlass_75_wmma_tensorop_f16_s...         0.00%       0.000us         0.00%       0.000us       0.000us       5.092ms         4.33%       5.092ms       5.092ms           0 b           0 b           0 b           0 b             1  
                                              aten::mul         0.25%       8.593ms         0.61%      20.639ms      43.726us       2.602ms         2.21%       2.626ms       5.564us           0 b           0 b      21.57 Mb      21.57 Mb           472  
                                            aten::copy_         0.13%       4.309ms         0.53%      17.970ms      44.813us       2.155ms         1.83%       2.176ms       5.426us           0 b           0 b           0 b           0 b           401  
     at::native::elementwise_kernel<128, 4, at::nati...         0.00%       0.000us         0.00%       0.000us       0.000us       1.797ms         1.53%       1.797ms       6.217us           0 b           0 b           0 b           0 b           289  
                                             aten::mean         0.09%       2.959ms         0.42%      14.193ms      97.880us       1.573ms         1.34%       1.607ms      11.086us           0 b           0 b      90.50 Kb      90.50 Kb           145  
     at::native::reduce_kernel<512, 1, at::native::R...         0.00%       0.000us         0.00%       0.000us       0.000us       1.573ms         1.34%       1.573ms      10.847us           0 b           0 b           0 b           0 b           145  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
- Self CPU time total: 3.392s
- Self CUDA time total: 117.495ms

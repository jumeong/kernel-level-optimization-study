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
- SeqLen: 256
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg       CPU Mem  Self CPU Mem      CUDA Mem  Self CUDA Mem    # of Calls  
                                     WQLinearMMFunction        59.22%        3.749s        59.82%        3.787s      15.027ms        1.620s        97.27%        1.632s       6.476ms           0 B           0 B     684.00 MB           0 B           252  
                                        awq_gemm_kernel         0.00%       0.000us         0.00%       0.000us       0.000us        1.620s        97.27%        1.620s       6.430ms           0 B           0 B           0 B           0 B           252  
                                  Lazy Function Loading         0.09%       5.566ms         0.09%       5.566ms      41.228us     752.622ms        45.18%     752.622ms       5.575ms           0 B           0 B           0 B           0 B           135  
                       Runtime Triggered Module Loading        18.49%        1.170s        18.49%        1.170s      25.445ms      55.793ms         3.35%      55.793ms       1.213ms           0 B           0 B           0 B           0 B            46  
                                              aten::mul         0.25%      15.586ms         8.67%     548.815ms       1.163ms      11.309ms         0.68%      14.007ms      29.677us           0 B           0 B       1.08 GB       1.08 GB           472  
                                            aten::copy_         0.14%       8.811ms         0.76%      48.416ms     120.739us       8.063ms         0.48%       8.114ms      20.235us           0 B           0 B           0 B           0 B           401  
                                               aten::mm         1.56%      98.795ms        12.42%     786.323ms     786.323ms       6.883ms         0.41%     805.339ms     805.339ms           0 B           0 B      74.19 MB      74.19 MB             1  
         turing_fp16_s1688gemm_fp16_256x128_ldg8_f2f_tn         0.00%       0.000us         0.00%       0.000us       0.000us       6.883ms         0.41%       6.883ms       6.883ms           0 B           0 B           0 B           0 B             1  
                                    Command Buffer Full        12.20%     772.377ms        12.20%     772.377ms       5.941ms       4.746ms         0.28%       4.746ms      36.509us           0 B           0 B           0 B           0 B           130  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
- Self CPU time total: 6.330s
- Self CUDA time total: 1.666s


## NSight Compute Profiling

### Summary
<img width="2608" height="291" alt="image" src="https://github.com/user-attachments/assets/07979a74-0883-4de1-b00b-140590393774" />

### Details
<img width="2585" height="186" alt="image" src="https://github.com/user-attachments/assets/369ef9b6-dd46-4ae7-a1be-e75c87557875" />

```python
a = tl.load(a_ptrs, mask=masks_a)
```
위 코드에서 Uncoalesced Shared Access가 많이 발생! 이에 따라, BLOCK_SIZE_K를 32에서 64로 변경해보기로

## Triton Debugging
- tl.static_print은 컴파일 시점에 tl.constexpr 변수의 값을 찍어준다.
- BLOCK_SIZE K: 32
- BLOCK_SIZE N: 32

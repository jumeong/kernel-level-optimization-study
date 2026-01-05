from vllm import LLM, SamplingParams
from vllm.v1.metrics.reader import Histogram
import torch, time, statistics

model_id = "Qwen/Qwen3-8B-AWQ"
#model_id = "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4"

llm = LLM(
        model=model_id,
        quantization="awq", # awq -> AutoAWQ, default -> awq-marlin
        disable_log_stats=False,
        #max_model_len=40960,
        )

params = SamplingParams(max_tokens=256, temperature=0)
prompt = "What is LLM?"

iters = 50
for _ in range(iters):
    _ = llm.generate([prompt], params)


for metric in llm.get_metrics():
    if isinstance(metric, Histogram):
        if metric.name == "vllm:time_to_first_token_seconds" or metric.name == "vllm:inter_token_latency_seconds":
            print(f"{metric.name}")
            print(f"Mean: {1000*metric.sum/metic.count} ms")

del llm

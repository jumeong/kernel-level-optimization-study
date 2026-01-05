import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import statistics

model_id = "Qwen/Qwen3-8B-AWQ"
device = "cuda"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map=device,
        torch_dtype=torch.float16
        )

prompt = "What is LLM?"
inputs = tokenizer([prompt], return_tensors="pt").to(device)

## TTFT Measurement
iters = 50
ttft = []
for _ in range(iters):
    torch.cuda.synchronize()
    start = time.time()

    model.generate(**inputs, max_new_tokens=1)

    torch.cuda.synchronize()
    end = time.time()

    ttft.append((end - start) * 1000)

## TPOT Measurement
tpot = []
for _ in range(iters):
    torch.cuda.synchronize()
    start = time.time()

    output = model.generate(**inputs, max_new_tokens=256)

    torch.cuda.synchronize()
    end = time.time()

    tpot.append(1000 * (end - start - statistics.mean(ttft)/1000) / (output.shape[-1] - inputs["input_ids"].shape[-1] - 1))
    

## statistics
ttft_mean = statistics.mean(ttft)
tpot_mean = statistics.mean(tpot)
print(f"TTFT mean: {ttft_mean:.2f} ms")
print(f"TPOT mean: {tpot_mean:.2f} ms")

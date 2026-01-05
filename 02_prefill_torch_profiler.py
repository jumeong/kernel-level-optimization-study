import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.profiler import profile, record_function, ProfilerActivity

model_id = "Qwen/Qwen3-8B-AWQ"
device = "cuda"

model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map=device,
        torch_dtype=torch.float16
        )

inputs = {                                                                  
        "input_ids": torch.randint(0, 2048, (1, 256), dtype=torch.int64).to(device),
        "attention_mask": torch.ones((1, 256), dtype=torch.int64).to(device)  
        }                                                                           
                                                                            
with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True
) as prof:
    model(**inputs)


print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=10))

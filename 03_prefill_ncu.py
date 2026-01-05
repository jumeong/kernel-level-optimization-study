import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.profiler import profile, record_function, ProfilerActivity

model_id = "Qwen/Qwen3-8B-AWQ"
device = "cuda"

#tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map=device,
        torch_dtype=torch.float16
        )

inputs = {
        "input_ids": torch.randint(0, 2048, (1, 256), dtype=torch.int64).to(device),
        "attention_mask": torch.ones((1, 256), dtype=torch.int64).to(device)
}

torch.cuda.profiler.start()
model(**inputs)
torch.cuda.profiler.stop()

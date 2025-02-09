import torch
print(f"Available GPUs: {torch.cuda.device_count()}")
# List memory usage for each GPU
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_properties(i).total_memory / 1024**2:.0f}MB total, "
      f"{torch.cuda.memory_allocated(i) / 1024**2:.0f}MB allocated")

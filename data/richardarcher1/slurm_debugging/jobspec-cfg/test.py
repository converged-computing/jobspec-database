import torch

print("="*120)
print("Torch was successfully loaded")
print("="*120)

print(f"{torch.__version__=}")
print("="*120)

try:
	print(f"{torch.cuda.is_available()=}")
except:
	print("Could not access cuda from torch")
print("="*120)

print("Investigate available devices:")
print("="*120)
devices = [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
print(devices)
print("="*120)

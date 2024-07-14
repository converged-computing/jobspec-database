"""
This file is used to debug in case of GPU(s) not being accessible.
"""

import torch

print("torch test")

torch.cuda.empty_cache()
print(torch.cuda.device_count())
print(torch.cuda.current_device())
# print(torch.cuda.device(0))
# print("Cuda is available: ", torch.cuda.is_available())
# print(torch.cuda.get_device_name(0))
# print(torch.cuda.get_device_name(1))
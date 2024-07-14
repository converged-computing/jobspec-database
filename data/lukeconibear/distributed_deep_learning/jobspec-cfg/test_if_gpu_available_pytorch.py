import torch

if torch.cuda.is_available():
    print(f'Yes, there are {torch.cuda.device_count()} GPUs available.')
else:
    print('No, GPUs are not available.')

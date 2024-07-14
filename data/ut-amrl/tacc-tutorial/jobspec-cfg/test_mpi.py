import os
import torch
import argparse

def main(args):
    print(f'Started job on TACC with id {args.job_id}')

    print('CUDA available:', torch.cuda.is_available())

    if torch.cuda.is_available():
        print('CUDA device count:', torch.cuda.device_count())
        print('CUDA device name:', torch.cuda.get_device_name(0))
        print('CUDA current device:', torch.cuda.current_device())
        device = f'cuda:{torch.cuda.current_device()}'
    else:
        device = "cpu"
    
    mat1 = torch.randn(3, 3).reshape(3, 3).to(device)
    mat2 = torch.randn(3, 4).reshape(3, 4).to(device)
    print("mat1:", mat1)
    print("mat2:", mat2)
    print("mat1 @ mat2", mat1 @ mat2)

    print(f'Finished job on TACC with id {args.job_id}')

if __name__ == '__main__':
    LAUNCHER_JID    = os.environ.get('LAUNCHER_JID')
    LAUNCHER_TSK_ID = os.environ.get('LAUNCHER_TSK_ID')

    parser = argparse.ArgumentParser()
    parser.add_argument('--job_id', type=int, default=0)
    args = parser.parse_args()

    main(args)

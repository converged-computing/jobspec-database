import os
import sys
import time
import torch
import argparse
import torch.distributed as dist
from functools import partial
from torch.profiler import profile, record_function, ProfilerActivity, schedule
from pathlib import Path
import itertools
from typing import Any
from enum import Enum, auto
from torch.cuda import Event
from typing import Tuple, Callable, Set, List, Optional


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--world_size",
        default=-1,
        type=int,
        help="number of nodes for distributed training",
    )
    parser.add_argument(
        "--global_rank",
        default=-1,
        type=int,
        help="global node rank for distributed training",
    )
    parser.add_argument(
        "--dist_url",
        default="env://",
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument(
        "--dist_backend", default="nccl", type=str, help="distributed backend"
    )
    parser.add_argument(
        "--local_rank", default=-1, type=int, help="local rank for distributed training"
    )
    parser.add_argument(
        "--job_dir",
        default="/n/home02/emyang/collective_benchmark",
        type=str,
        help="job directory",
    )
    parser.add_argument(
        "--out_dir",
        default="/n/home02/emyang/collective_benchmark/bandwidth_benchmark",
        type=str,
        help="output directory for benchmarking results",
    )
    args = parser.parse_args()

    return args


def print_env():
    print("World Size: ", os.environ["WORLD_SIZE"])
    print("Master Addr: ", os.environ["MASTER_ADDR"])
    print("Master Port:", os.environ["MASTER_PORT"])
    print("Slurm Procid: ", os.environ["SLURM_PROCID"])


class Task:
    def __init__(self) -> None:
        pass

    def __call__(self, args: Any) -> Any:

        print_env()
        print(args)

        if "WORLD_SIZE" in os.environ:
            args.world_size = int(os.environ["WORLD_SIZE"])
            self.world_size = args.world_size

        args.distributed = args.world_size > 1
        ngpus_per_node = torch.cuda.device_count()

        if "SLURM_PROCID" in os.environ:
            args.global_rank = int(os.environ["SLURM_PROCID"])
            args.local_rank = args.global_rank % ngpus_per_node

            dist.init_process_group(
                backend=args.dist_backend,
                init_method=args.dist_url,
                world_size=args.world_size,
                rank=args.global_rank,
            )

            torch.cuda.set_device(args.local_rank)

        self.experiment(args)
        return "Success"

    def experiment(self, args):
        Path(args.out_dir).mkdir(parents=True, exist_ok=True)
        out_file = f"{args.out_dir}/bw_{args.global_rank}.data"
        fout = open(out_file, "w")

        dist.barrier()

        discard_iters = 2

        current_size = 0
        size = 256 * (2**10)  # Initial total size is 256 KB

        for _ in range(discard_iters):
            self.send_recv_bench(args, size)

        # Measure latency by sending a 1-element tensor
        latency = self.send_recv_bench(args, 1)
        fout.write(f"0, {latency}\n")

        # Construct data size range for benchmarking
        data_sizes = []
        # Exponential data sizes
        for i in range(6, 29):
            data_sizes.append(2**i)

        # Additional data sizes
        for i in range(34):
            if i == 2:
                size = 512 * (2**10) # increments of 256 KB
            elif i == 11:
                size = 1 * (2**20) # increments of 1 MB
            elif i == 26:
                size = 10 * (2**20) # increments of 10 MB
            current_size += size
            if current_size not in data_sizes:
                data_sizes.append(current_size)

        for size in data_sizes:
            size_in_mb = size / 2**20

            time = self.send_recv_bench(args, size // 4)
            fout.write(f"{size_in_mb}, {time}\n")

            dist.barrier()

        fout.close()

    def send_recv_bench(self, args, data_size):

        tensor = torch.randn(data_size, device=torch.device("cuda"))
        in_tensor = torch.empty(data_size, device=torch.device("cuda"))

        dist.barrier()

        src_rank = 0
        dst_rank = 1

        # Average over three trials
        niters = 3
        times = [0 for _ in range(niters)]

        for i in range(niters):
            if src_rank == dist.get_rank():
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                dist.send(tensor, dst=dst_rank)
                end.record()
            elif dst_rank == dist.get_rank():
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                dist.recv(in_tensor, src=src_rank)
                end.record()

            torch.cuda.synchronize()
            time = start.elapsed_time(end)
            times[i] = time

            dist.barrier()

        return sum(times)/niters


if __name__ == "__main__":
    args = parse_args()
    task = Task()
    task(args)

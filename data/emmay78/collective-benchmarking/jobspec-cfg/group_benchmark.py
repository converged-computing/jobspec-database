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


class Collective(Enum):
    all_reduce = "all_reduce"
    all_gather = "all_gather"
    reduce_scatter = "reduce_scatter"

    def __str__(self) -> str:
        return self.value


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--collective",
        default=Collective.all_gather,
        type=Collective,
        choices=list(Collective),
        help="collective function to benchmark [all_reduce, broadcast, reduce, scatter, gather, all_gather, all_to_all, reduce_scatter]",
    )
    parser.add_argument(
        "--coalesce",
        default=False,
        type=bool,
        help="Use coalescing manager. Default is False.",
    )
    parser.add_argument(
        "--num_to_coalesce",
        default=4,
        type=int,
        help="Number of tensorsto coalesce",
    )
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
        default="/n/home02/emyang/collective_benchmark/coalescing_benchmark_results",
        type=str,
        help="output directory for benchmarking results",
    )
    parser.add_argument(
        "--data_size",
        default=5 * (2**18),
        type=int,
        help="Data size for profile size.",
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

    def rank0_print(self, rank: int, *args, **kwargs):
        if rank == 0:
            print(*args, **kwargs, flush=True)

    def get_collective_function(self, collective_to_benchmark: Collective):
        if collective_to_benchmark == Collective.all_reduce:
            return dist.all_reduce
        elif collective_to_benchmark == Collective.reduce_scatter:
            return dist.reduce_scatter_tensor
        elif collective_to_benchmark == Collective.all_gather:
            return dist.all_gather_into_tensor

    def experiment(self, args):
        dist.barrier()
        rank = dist.get_rank()

        name = f"{args.collective.__str__()}_{args.world_size}_{dist.get_rank()}"
        delay_dir = f"{args.out_dir}/{args.collective.__str__()}/"
        if args.coalesce:
            delay_dir += "/coalesce/"
        else:
            delay_dir += "/default/"
        Path(delay_dir).mkdir(parents=True, exist_ok=True)
        data_file = f"{delay_dir}/{name}"
        fout = open(f"{data_file}.data", "w")

        size = 5 * (2**18)  # Initial total size is 5 MB
        per_rank_size = torch.Size([size // self.world_size])
        dest_numel = per_rank_size.numel() * self.world_size
        dest_tensor_ref = torch.randn((dest_numel,), device=torch.cuda.current_device())
        dest_tensor = torch.empty((dest_numel,), device=torch.cuda.current_device())

        niters = 10
        discard_iters = 2

        current_size = 0
        for i in range(45):
            if i == 20:
                size = 20 * (2**18)
            elif i == 30:
                size = 50 * (2**18)
            current_size += size

            per_rank_size = torch.Size([current_size // self.world_size])
            dest_numel = per_rank_size.numel() * self.world_size
            dest_tensor_ref = torch.randn(
                (dest_numel,), device=torch.cuda.current_device()
            )
            dest_tensor = torch.empty((dest_numel,), device=torch.cuda.current_device())

            elapsed_times = []

            for _ in range(niters + discard_iters):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()

                if args.coalesce:
                    self.experiment_coalesce(args, dest_tensor, dest_tensor_ref)
                else:
                    self.experiment_default(args, dest_tensor, dest_tensor_ref)

                end.record()
                torch.cuda.synchronize()

                dest_tensor.zero_()

                elapsed_time = start.elapsed_time(end)
                elapsed_times.append(elapsed_time)

            time_per_call = sum(elapsed_times[discard_iters:]) / len(
                elapsed_times[discard_iters:]
            )

            self.rank0_print(
                rank,
                f"[Rank {rank}] time / coalesced all-gather ({args.num_to_coalesce} coalesced): {time_per_call:.5f} ms",
            )

            size_in_mb = (dest_numel * 4) // 2**20
            # write results
            for delay in elapsed_times[discard_iters:]:
                fout.write(f"{size_in_mb}, {delay:.4f}\n")

            dist.barrier(device_ids=[args.local_rank])

        fout.close()

    def coalesce_inputs(
        self,
        collective: Collective,
        dest_tensor: torch.Tensor,
        dest_tensor_ref: torch.Tensor,
        num_to_coalesce: int,
        coalesce_iter: int,
    ):
        rank = dist.get_rank()
        outer_incr = dest_tensor_ref.size(dim=0) // num_to_coalesce
        inner_incr = outer_incr // args.world_size

        dest_offset = outer_incr * coalesce_iter
        if args.collective == Collective.all_reduce:
            src_tensor = dest_tensor_ref[
                dest_offset + rank * inner_incr : dest_offset + (rank + 1) * inner_incr
            ]
            return (src_tensor,)
        else:
            dest_tensor_i = dest_tensor[dest_offset : dest_offset + outer_incr]
            src_tensor = dest_tensor_ref[
                dest_offset + rank * inner_incr : dest_offset + (rank + 1) * inner_incr
            ]
            if args.collective == Collective.all_gather:
                return (dest_tensor_i, src_tensor)
            else:
                return (src_tensor, dest_tensor_i)

    def experiment_coalesce(
        self, args, dest_tensor: torch.Tensor, dest_tensor_ref: torch.Tensor
    ):
        collective_function = self.get_collective_function(args.collective)

        from torch.distributed.distributed_c10d import _coalescing_manager

        with _coalescing_manager(group=None, device=torch.device("cuda"), async_ops=True) as cm:
            for i in range(args.num_to_coalesce):
                input_args = self.coalesce_inputs(
                    args.collective,
                    dest_tensor,
                    dest_tensor_ref,
                    args.num_to_coalesce,
                    i,
                )
                ret = collective_function(*input_args, async_op=True)
        cm.wait()

    def default_inputs(
        self,
        collective: Collective,
        dest_tensor: torch.Tensor,
        dest_tensor_ref: torch.Tensor,
    ):
        rank = dist.get_rank()
        offsets = [0] + list(
            itertools.accumulate(
                [
                    dest_tensor_ref.size(dim=0) // self.world_size
                    for _ in range(self.world_size)
                ]
            )
        )
        src_tensor = dest_tensor_ref[offsets[rank] : offsets[rank + 1]]

        if args.collective == Collective.all_reduce:
            return (src_tensor,)
        elif args.collective == Collective.all_gather:
            return (dest_tensor, src_tensor)
        elif args.collective == Collective.reduce_scatter:
            return (src_tensor, dest_tensor)

    def experiment_default(
        self, args, dest_tensor: torch.Tensor, dest_tensor_ref: torch.Tensor
    ):
        collective_function = self.get_collective_function(args.collective)
        input_args = self.default_inputs(args.collective, dest_tensor, dest_tensor_ref)
        collective_function(*input_args)


if __name__ == "__main__":
    args = parse_args()
    task = Task()
    task(args)

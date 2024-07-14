import os
import sys
import time
import torch
import argparse
import torch.distributed as dist
from functools import partial
from torch.profiler import profile, record_function, ProfilerActivity, schedule
from pathlib import Path
from typing import Any
from enum import Enum, auto
from torch.cuda import Event
from typing import Tuple, Callable


class Collective(Enum):
    all_reduce = "all_reduce"
    broadcast = "broadcast"
    reduce = "reduce"
    all_gather = "all_gather"
    gather = "gather"
    scatter = "scatter"
    reduce_scatter = "reduce_scatter"
    all_to_all = "all_to_all"

    def __str__(self) -> str:
        return self.value


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--collective",
        default=Collective.all_reduce,
        type=Collective,
        choices=list(Collective),
        help="collective function to benchmark [all_reduce, broadcast, reduce, scatter, gather, all_gather, all_to_all, reduce_scatter]",
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
        default="/n/home02/emyang/collective_benchmark/benchmark_results",
        type=str,
        help="output directory for benchmarking results",
    )
    parser.add_argument(
        "--profile",
        default=False,
        type=bool,
        help="Measure with PyTorch Profiler. Disabled by default.",
    )
    parser.add_argument(
        "--async_op",
        default=False,
        type=bool,
        help="Benchmark using an asynchronous collective operation. The collective operation function returns a distributed request object on which wait() is called to block the process until completion.",
    )
    parser.add_argument(
        "--profile_size",
        default=5 * (2 ** 18),
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
    async_collective = None

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
            if args.profile:
                self.profile(args)
            else:
                self.experiment(args)
        return "Success"

    # If the asynchronous operation for the collective is specified,
    # run the collective with async_op = True and block the current process
    # until completion
    def collective_wait(self, *input_args):
        handle = self.async_collective(*input_args, async_op=True)
        handle.wait()

    def get_collective_function(
        self, collective_to_benchmark: Collective, async_op: bool
    ) -> Callable:
        if not async_op:
            if collective_to_benchmark == Collective.all_reduce:
                return dist.all_reduce
            elif collective_to_benchmark == Collective.reduce_scatter:
                return dist.reduce_scatter_tensor
            elif collective_to_benchmark == Collective.all_to_all:
                return dist.all_to_all
            elif collective_to_benchmark == Collective.broadcast:
                return dist.broadcast
            elif collective_to_benchmark == Collective.reduce:
                return dist.reduce
            elif collective_to_benchmark == Collective.all_gather:
                return dist.all_gather_into_tensor
            elif collective_to_benchmark == Collective.gather:
                return dist.gather
        else:
            self.async_collective = self.get_collective_function(
                collective_to_benchmark, async_op=False
            )
            return self.collective_wait

    def create_tensors_all_reduce(self, size: Tuple[int, ...]) -> Tuple[torch.Tensor]:
        tensor = (
            torch.arange(size, dtype=torch.float32, device=torch.cuda.current_device())
            + dist.get_rank() * size
        )
        return (tensor,)

    def create_tensors_reduce_scatter(
        self, size: Tuple[int, ...]
    ) -> Tuple[torch.Tensor]:
        tensor_in = (
            torch.arange(
                size * self.world_size,
                dtype=torch.float32,
                device=torch.cuda.current_device(),
            )
            + size * self.world_size * dist.get_rank()
        )

        tensor_out = torch.zeros(
            size, dtype=torch.float32, device=torch.cuda.current_device()
        )

        return (tensor_out, tensor_in)

    def create_tensors_all_to_all(self, size: Tuple[int, ...]) -> Tuple[torch.Tensor]:
        tensor_in = (
            torch.arange(size * self.world_size, device=torch.cuda.current_device())
            + dist.get_rank() * size * self.world_size
        )

        tensor_in = list(tensor_in.chunk(self.world_size))
        tensor_out = list(
            torch.empty(
                [size * self.world_size],
                dtype=torch.float32,
                device=torch.cuda.current_device(),
            ).chunk(self.world_size)
        )
        return (tensor_out, tensor_in)

    def create_tensors_broadcast(self, size: Tuple[int, ...]) -> Tuple[torch.Tensor]:
        if dist.get_rank() == 0:
            return (torch.randn(size, device=torch.cuda.current_device()), 0)
        else:
            return (torch.empty([size], device=torch.cuda.current_device()), 0)

    def create_tensors_reduce(self, size: Tuple[int, ...]) -> Tuple[torch.Tensor]:
        return (torch.randn(size, device=torch.cuda.current_device()), 0)

    def create_tensors_all_gather(self, size: Tuple[int, ...]) -> Tuple[torch.Tensor]:
        tensor_out = torch.zeros(
            size * self.world_size,
            dtype=torch.float32,
            device=torch.cuda.current_device(),
        )
        tensor_in = (
            torch.arange(size, dtype=torch.float32, device=torch.cuda.current_device())
            + size * dist.get_rank()
        )
        return (tensor_out, tensor_in)

    def create_tensors_gather(self, size: Tuple[int, ...]) -> Tuple[torch.Tensor]:
        tensor = (
            torch.arange(size, dtype=torch.float32, device=torch.cuda.current_device())
            + 1
            + size * self.world_size * dist.get_rank()
        )
        gather_list = (
            [
                torch.empty(
                    [size], dtype=torch.float32, device=torch.cuda.current_device()
                )
                for _ in range(self.world_size)
            ]
            if dist.get_rank() == 0
            else None
        )
        return (tensor, gather_list, 0)

    def get_create_tensor_function(
        self, collective_to_benchmark: Collective
    ) -> Callable:
        if collective_to_benchmark == Collective.all_reduce:
            return self.create_tensors_all_reduce
        elif collective_to_benchmark == Collective.reduce_scatter:
            return self.create_tensors_reduce_scatter
        elif collective_to_benchmark == Collective.all_to_all:
            return self.create_tensors_all_to_all
        elif collective_to_benchmark == Collective.broadcast:
            return self.create_tensors_broadcast
        elif collective_to_benchmark == Collective.reduce:
            return self.create_tensors_reduce
        elif collective_to_benchmark == Collective.all_gather:
            return self.create_tensors_all_gather
        elif collective_to_benchmark == Collective.gather:
            return self.create_tensors_gather

    def get_number_of_tensors(self, collective_to_benchmark: Collective) -> int:
        if collective_to_benchmark == Collective.all_reduce:
            return 1
        elif collective_to_benchmark == Collective.reduce_scatter:
            return 2
        elif collective_to_benchmark == Collective.all_to_all:
            return 2
        elif collective_to_benchmark == Collective.broadcast:
            return 1
        elif collective_to_benchmark == Collective.reduce:
            return 1
        elif collective_to_benchmark == Collective.all_gather:
            return self.world_size + 1
        elif collective_to_benchmark == Collective.gather:
            return self.world_size + 1

    def experiment(self, args):
        # Get total memory available on CUDA device
        total_mem = torch.cuda.get_device_properties(0).total_memory
        total_mem -= 2 * (2 ** 30)  # subtract 2 GB

        collective_function = self.get_collective_function(
            args.collective, async_op=args.async_op
        )
        create_args_function = self.get_create_tensor_function(args.collective)

        warmup_iters = 10
        niters = 10
        size = 5 * (2 ** 18)  # Initial size is 5 MB

        current_size = 0
        num_tasks = os.environ["WORLD_SIZE"]
        name = args.collective.__str__() + f"_{num_tasks}_{dist.get_rank()}"
        delay_dir = f"{args.out_dir}/{args.collective.__str__()}_{args.world_size}"
        Path(delay_dir).mkdir(parents=True, exist_ok=True)
        data_file = f"{delay_dir}/{name}"
        if args.async_op:
            data_file += "_async"
        fout = open(f"{data_file}.data", "w")

        ######################################
        # 1. warming up CUDACachingAllocator #
        ######################################
        for _ in range(warmup_iters):
            try:
                input_args = create_args_function(size)
            except torch.cuda.OutOfMemoryError:
                print("Ran out of CUDA memory during warm-up")
            else:
                collective_function(*input_args)

        # Construct data size range for benchmarking
        data_sizes = []
        # Exponential data sizes
        for i in range(6, 29):
            data_sizes.append(2 ** i)

        # Additional data sizes
        for i in range(44):
            if i == 2:
                size = 512 * (2 ** 10)  # increments of 256 KB
            elif i == 11:
                size = 1 * (2 ** 20)  # increments of 1 MB
            elif i == 26:
                size = 10 * (2 ** 20)  # increments of 10 MB
            elif i == 35:
                size = 100 * (2 ** 20)  # increments of 100 MB
            current_size += size
            if current_size not in data_sizes:
                data_sizes.append(current_size)

        for size in data_sizes:
            size_in_mb = size / 2 ** 20

            if (current_size * 4) > (
                total_mem // get_number_of_tensors(args.collective)
            ):
                break

            ##################################################################
            # 2. measure raw delays and memory to rule out profiler overhead #
            ##################################################################
            if i == 0:
                niters += 2

            events_pre = [Event(enable_timing=True) for _ in range(niters)]
            events_post = [Event(enable_timing=True) for _ in range(niters)]

            for experiment_idx in range(niters):
                try:
                    input_args = create_args_function(current_size)
                except torch.cuda.OutOfMemoryError:
                    print(
                        "Ran out of CUDA memory creating tensor of size", current_size
                    )
                else:
                    events_pre[experiment_idx].record()
                    collective_function(*input_args)
                    events_post[experiment_idx].record()

            torch.cuda.synchronize()

            delays = [
                pre.elapsed_time(post) for pre, post in zip(events_pre, events_post)
            ]

            # The first experiment has a much larger CUDA time than all other experiments.
            # Thus, we discard the first two measurements.
            if i == 0:
                delays = delays[2:]
                niters -= 2

            # write results
            for delay in delays:
                fout.write(f"{size_in_mb}, {delay:.4f}\n")

            # wait for all peers to finish
            dist.barrier(device_ids=[args.local_rank])

        fout.close()
        self.teardown()
        return {
            "data_size": size_in_mb,
        }

    def profile(self, args):
        collective_function = self.get_collective_function(
            args.collective, async_op=args.async_op
        )
        create_args_function = self.get_create_tensor_function(args.collective)

        num_tasks = os.environ["WORLD_SIZE"]
        name = args.collective.__str__() + f"_{num_tasks}_{dist.get_rank()}"
        delay_dir = f"{args.out_dir}/{args.collective.__str__()}_{args.world_size}"
        Path(delay_dir).mkdir(parents=True, exist_ok=True)
        profile_file = f"{delay_dir}/{name}"
        if args.async_op:
            profile_file += "_async"
        profile_fout = open(f"{profile_file}.profiler.data", "w")

        schedule = torch.profiler.schedule(wait=1, warmup=5, active=10,)

        try:
            input_args = create_args_function(args.profile_size)
        except torch.cuda.OutOfMemoryError:
            print("Ran out of CUDA memory creating tensor of size", args.profile_size)
        else:
            with profile(
                activities=[ProfilerActivity.CUDA],
                record_shapes=True,
                profile_memory=True,
                schedule=schedule,
            ) as prof:
                for _ in range(15):
                    collective_function(*input_args)
                    prof.step()

        profile_fout.write(
            prof.key_averages().table(sort_by="cuda_time_total", row_limit=10)
        )

        profile_fout.close()
        self.teardown()
        size_in_mb = (args.profile_size * 4) // 2 ** 20
        return {"data_size": size_in_mb}

    def teardown(self):
        dist.destroy_process_group()


if __name__ == "__main__":
    args = parse_args()
    task = Task()
    task(args)

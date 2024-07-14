#!/usr/bin/env python

import argparse
import numpy as np

from math import ceil
from mpi4py import MPI
from timeit import default_timer as timer
from utils import calculate_interactions

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('N', type=int)
    parser.add_argument('zad', type=int, choices=[1, 2, 3])

    args = parser.parse_args()

    N = args.N
    zad = args.zad

    steps = 20 if zad == 3 else 1
    dt = 10

    start_time = timer()

    comm = MPI.COMM_WORLD
    p = comm.Get_size()
    rank = comm.Get_rank()

    N = ceil(N / p) * p

    assert N % p == 0  # TODO: support number of stars indivisible by number of processes
    n = N // p

    left = (rank - 1) % p
    right = (rank + 1) % p

    # Każdy z procesów zawiera równoliczny zbiór gwiazd N/p, ...
    start = rank * n
    end = (rank + 1) * n
    stars_own = np.load('stars.npy')[start:end, :]  # M, x, y, z, vx, vy, yz

    if zad == 3:
        snapshots_own = np.empty((steps, N // p, 3))  # x, y, z
        a_prev = np.zeros((n, 3))

    for step in range(steps):

        # ...bufor na gwiazdy przechodnie...
        # Początkowo w buforze przechodnim znajdują się własne gwazdy danego procesu.
        stars_buffer = np.concatenate([stars_own[:, :4]] * 2)  # M, x, y, z + double buffering
        stars_buffer_A = stars_buffer[:n]
        stars_buffer_B = stars_buffer[n:]

        # ...bufor na akumulator przechodni...
        if zad >= 2:
            accumulator_buffer = np.zeros((n * 2, 3))
            accumulator_buffer_A = accumulator_buffer[:n]
            accumulator_buffer_B = accumulator_buffer[n:]

        # ...oraz akumulator swoich interakcji.
        interactions = calculate_interactions(stars_own[:, :4], stars_own[:, :4], same=True, use_symmetry=zad >= 2)
        masses_own = stars_own[:, 0, np.newaxis]
        accumulator = (interactions * masses_own).sum(axis=1)  # ax, ay, az

        # Każdy z procesów powtarza p-1 razy następujące czynności:
        # Następnie powtarza floor(p/2) razy następujące czynności:
        iters = p // 2 if zad >= 2 else p-1
        for i in range(iters):

            # 1. przesyła przechodnią porcję(N/p) gwiazd do lewego sąsiada w pierścieniu procesów
            comm.Isend([stars_buffer_A, MPI.FLOAT], dest=left)
            if zad >= 2:
                # 1. przesyła do lewego sąsiada przechodnią porcję (N/p) gwiazd oraz skojarzony z nimi akumulator przechodni.
                comm.Isend([accumulator_buffer_A, MPI.FLOAT], dest=left)

            # 2. odbiera porcję (N/p) gwiazd od prawego sąsiada do swojego bufora przechodniego
            comm.Recv([stars_buffer_B, MPI.FLOAT], source=right)
            if zad >= 2:
                # 2. odbiera porcję (N/p) gwiazd oraz skojarzony z nimi akumulator przechodni od prawego sąsiada do swoich buforów
                comm.Recv([accumulator_buffer_B, MPI.FLOAT], source=right)

            # 3. oblicza interakcje swoich gwiazd i gwiazd otrzymanych
            interactions = calculate_interactions(stars_own[:, :4], stars_buffer_B, same=False)

            # 4. dodaje obliczona interakcje do swojego akumulatora
            masses_buffer = stars_buffer_B[:, 0, np.newaxis]
            accumulator += (interactions * masses_buffer).sum(axis=1)

            if zad >= 2 and not (p % 2 == 0 and i == (p // 2) - 1):
                # 5. dodaje obliczone interakcje do bufora przechodniego
                accumulator_buffer_B += (-interactions.transpose(1, 0, 2) * masses_own).sum(axis=1)

            # swap buffers
            stars_buffer_A, stars_buffer_B = stars_buffer_B, stars_buffer_A
            if zad >= 2:
                accumulator_buffer_A, accumulator_buffer_B = accumulator_buffer_B, accumulator_buffer_A

        if zad >= 2:
            # Po ostatnim kroku wysyła akumulator przechodni do właściciela gwiazd ostatniej przechodniej porcji.
            comm.Isend([accumulator_buffer_A, MPI.FLOAT], dest=(rank - (p+1) // 2) % p)
            comm.Recv([accumulator_buffer_B, MPI.FLOAT], source=(rank + (p+1) // 2) % p)
            accumulator += accumulator_buffer_B

        if zad == 3:
            x = stars_own[:, 1:4]
            v = stars_own[:, 4:]

            x += v * dt + a_prev * dt * dt / 2
            v += (accumulator + a_prev) * dt / 2

            snapshots_own[step, :, :] = x

            a_prev = accumulator

    # zebranie danych
    if zad <= 2:
        accelerations = np.empty((p, N // p, 3)) if rank == 0 else None
        comm.Gather(sendbuf=accumulator, recvbuf=accelerations)
    else:
        snapshots = np.empty((p, steps, N // p, 3)) if rank == 0 else None
        comm.Gather(sendbuf=snapshots_own, recvbuf=snapshots)

    if rank == 0:  # root
        if zad <= 2:
            accelerations = accelerations.reshape(N, 3)
            # np.save(f'accelerations_zad{zad}_np{p}.npy', accelerations)
        else:
            snapshots = snapshots.transpose(1, 0, 2, 3).reshape(steps, N, 3)
            # np.save('snapshots.npy', snapshots)

        duration = timer() - start_time
        print(f'{N},{duration}')

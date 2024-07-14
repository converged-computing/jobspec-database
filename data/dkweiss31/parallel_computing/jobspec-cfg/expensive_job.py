import argparse
import scipy as sp
import numpy as np


def wow_much_expensive_matrixexp(N: int, r: int):
    print(f"matrix of size ({N}, {N}) with rng {r}")
    rng = np.random.default_rng(r)
    mat = rng.standard_normal((N, N))
    return sp.linalg.expm(mat)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Expensive job to run in parallel!')
    parser.add_argument("--N", default=1000, type=int, help="matrix size")
    parser.add_argument("--r", default=1, type=int, help="seed for rng")
    args = parser.parse_args()
    wow_much_expensive_matrixexp(args.N, args.r)

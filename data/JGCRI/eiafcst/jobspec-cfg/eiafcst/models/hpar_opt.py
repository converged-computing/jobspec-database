"""
Optimize model hyperparameters using random search.

Caleb Braun
5/30/19
"""
import numpy as np
import sys

from eiafcst.models.model_gdp import run


class ArgBuilder:
    """Generate a simple class containing random argument values."""

    def __init__(self, id, r_lr, r_Cn, r_Ck, r_Cf, r_L1, r_L2, r_lgdp1, r_lgdp2):
        """Randomly sample from input hyperparameter ranges."""
        self.lr = np.random.choice(r_lr)

        self.C = self.build_C_arg(r_Cn, r_Ck, r_Cf)

        self.L1 = np.random.choice(r_L1)
        self.L2 = np.random.choice(r_L2)
        self.lgdp1 = np.random.choice(r_lgdp1)
        self.lgdp2 = np.random.choice(r_lgdp2)

        # We now train these two outputs separately
        self.wgdp = 0
        self.wdec = 1

        self.epochs = 10000
        self.patience = 400

    @staticmethod
    def factors(n):
        """Get the factors of a number."""
        return np.array(list(set(i for i in range(1, int(n**0.5) + 1) if not n % i)))

    def build_C_arg(self, r_Cn, r_Ck, r_Cf):
        """Build the string argument for the convolutional layers."""
        width = 168  # the starting width of the first layer
        c = []
        for i in range(np.random.choice(r_Cn)):
            r_Ck = r_Ck[r_Ck < width]  # Keep the kernel size smaller than the layer length
            Ck = np.random.choice(r_Ck)
            Cf = np.random.choice(r_Cf)
            pool_size = np.random.choice(self.factors(width)[1:])  # Random factor, but not 1
            Cp = pool_size

            c.append('-'.join([str(p) for p in [Ck, Cf, Cp]]))

            width /= pool_size
            if len(self.factors(width)) == 1:
                break

        c = ','.join(c)
        print(c)
        return c


def optimize(n, repeat=3, out='gdp_results.csv'):
    """
    Optimize parameters.

    :param n:       How many variations to run (int)
    :param repeat:  How many times to repeat each variation
    :param out:     File name to store results in

    The hyperparameters we need to optimize are:
        lr - Learning Rate
        C - Convolutional layers
        L1 - Hidden layer after convolutional layers
        L2 - Final encoded layer, represents features from electricity dataset
        lgdp - Hidden layer in GDP branch
    """
    r_lr = np.array([0.001])

    # ranges for the convolutional layers (n: how many, k: kernel size, f: filter size, p: pool size)
    r_Cn = np.arange(1, 3 + 1)
    r_Ck = np.arange(2, 48 + 1)
    r_Cf = np.arange(1, 24 + 1)

    r_L1 = np.arange(1, 24 + 1)
    r_L2 = np.arange(1, 16 + 1)
    r_lgdp1 = np.arange(1, 12 + 1)
    r_lgdp2 = np.arange(0, 12 + 1)

    for i in range(n):
        args = ArgBuilder(i, r_lr, r_Cn, r_Ck, r_Cf, r_L1, r_L2, r_lgdp1, r_lgdp2)
        for j in range(repeat):
            args.model = f'eiafcst/models/diagnostic/gdp/{out.split(".")[0]}_{i}-{j}.h5'
            run(args, out)


if __name__ == '__main__':
    try:
        n = int(sys.argv[1])
    except IndexError:
        raise "Please provide how many random searches to run."
    except ValueError:
        raise "Please provide an integer for how many random searches to run."

    try:
        repeat = int(sys.argv[2])
    except IndexError:
        raise "Please provide how many times to repeat each search."
    except ValueError:
        raise "Please provide an integer for how many times to repeat each search."

    try:
        out = sys.argv[3]
    except IndexError:
        raise "Please provide an output file name."

    optimize(n, repeat, out)

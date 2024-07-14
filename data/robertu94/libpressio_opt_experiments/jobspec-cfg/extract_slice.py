#!/usr/bin/env python

import argparse
import numpy as np

TYPES = {
    "float": np.float32,
    "double": np.float64,
    "int8": np.int8,
    "int16": np.int16,
    "int32": np.int32,
    "int64": np.int64,
    "uint8": np.uint8,
    "uint16": np.uint16,
    "uint32": np.uint32,
    "uint64": np.uint64,
}


def to_type(x: str) -> np.dtype:
    "return a numpy type from a string"""
    return TYPES[x]


parser = argparse.ArgumentParser()
parser.add_argument("--api", type=int)
parser.add_argument("--config")
parser.add_argument("--input")
parser.add_argument("--decompressed")
parser.add_argument("--type", type=to_type)
parser.add_argument("--dims", type=int, action='append', default=list())
parser.add_argument("--external_outfile", dest="outfile")
parser.add_argument("--external_from_slice", dest="from_slice", type=int,  default=0)
parser.add_argument("--external_to_slice", dest="to_slice", type=int, default=1)
parser.add_argument("--external_slice_dim", dest="slice_dim", type=int, default=0)
parser.add_argument("--external_visualize", dest="visualize", action="store_true")
args = parser.parse_args()

slices = tuple(
    slice(args.from_slice, args.to_slice) if i == args.slice_dim else slice(None, None)
    for i in range(len(args.dims)))

print("external:api=5")
if args.input:
    data = np.fromfile(args.input, dtype=args.type)
    data = data.reshape(*args.dims)
    data = data[slices]

    if args.visualize:
        import matplotlib.pyplot as plt
        midpoint = (args.to_slice - args.from_slice)//2
        vis_slice = tuple(
                midpoint if i == args.slice_dim else slice(None, None)
                for i in range(len(args.dims))
                )
        plt.imshow(data[vis_slice])
        plt.show()

    data.tofile(args.outfile)
